#!/usr/bin/env python3
"""
Retrain DeepLabCut from manual_labels exports and optionally rerun PreyTouch on one video.
"""

import argparse
import csv
import importlib
import importlib.util
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def log(message: str):
    print(message, flush=True)


def require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for retraining. Install with: pip install pyyaml"
        ) from exc
    return yaml


def require_deeplabcut():
    try:
        import deeplabcut  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "deeplabcut":
            raise
        raise RuntimeError(
            "DeepLabCut is required for retraining. Install in the active environment."
        ) from exc
    return deeplabcut


def load_dlc_project_config(config_path: Path):
    yaml = require_yaml()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config yaml: {config_path}")
    project_path_text = cfg.get("project_path")
    project_path = Path(project_path_text).expanduser().resolve() if project_path_text else config_path.parent.resolve()
    scorer = str(cfg.get("scorer", "manual"))
    bodyparts = [str(bp) for bp in cfg.get("bodyparts", [])]
    if len(bodyparts) == 0:
        raise RuntimeError(f"No bodyparts found in config: {config_path}")
    return cfg, project_path, scorer, bodyparts


def save_dlc_project_config(config_path: Path, cfg: dict):
    yaml = require_yaml()
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_manual_label_file(label_path: Path) -> Dict[str, Tuple[float, float]]:
    with open(label_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not raw_lines:
        return {}

    reader = csv.DictReader(raw_lines)
    points: Dict[str, Tuple[float, float]] = {}
    for row in reader:
        point_name = (row.get("point_name") or "").strip()
        if not point_name:
            continue

        x_text = row.get("x_pixel") or row.get("x_raw")
        y_text = row.get("y_pixel") or row.get("y_raw")
        if x_text is None or y_text is None:
            continue

        try:
            x_val = float(x_text)
            y_val = float(y_text)
        except Exception:
            continue

        points[point_name] = (x_val, y_val)
    return points


def map_point_name(point_name: str, bodyparts_set: set) -> Optional[str]:
    candidates = [point_name]
    if point_name.endswith("_cam"):
        candidates.append(point_name[:-4])
    else:
        candidates.append(f"{point_name}_cam")

    for candidate in candidates:
        if candidate in bodyparts_set:
            return candidate
    return None


def find_matching_image(images_dir: Path, stem: str) -> Optional[Path]:
    for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = images_dir / f"{stem}{suffix}"
        if img.exists():
            return img
    return None


def _safe_tag(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)


def flatten_manual_labels_root(labels_root: Path) -> Dict[str, int]:
    """Merge legacy per-video export folders into one shared labels_root/images+labels pool."""
    target_images = labels_root / "images" / "train"
    target_labels = labels_root / "labels" / "train"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    moved_pairs = 0
    skipped_no_image = 0

    try:
        target_labels_resolved = target_labels.resolve()
    except Exception:
        target_labels_resolved = target_labels
    try:
        labels_root_resolved = labels_root.resolve()
    except Exception:
        labels_root_resolved = labels_root

    source_sets = []
    for labels_dir in sorted(labels_root.rglob("labels/train")):
        try:
            labels_dir_resolved = labels_dir.resolve()
        except Exception:
            labels_dir_resolved = labels_dir
        if labels_dir_resolved == target_labels_resolved:
            continue

        source_root = labels_dir.parent.parent
        try:
            source_root_resolved = source_root.resolve()
        except Exception:
            source_root_resolved = source_root
        if source_root_resolved == labels_root_resolved:
            continue

        images_dir = source_root / "images" / "train"
        if images_dir.exists():
            source_sets.append((source_root, images_dir, labels_dir))

    for source_root, source_images, source_labels in source_sets:
        source_tag = _safe_tag(source_root.name or "source")

        for label_file in sorted(source_labels.glob("*.txt")):
            image_file = find_matching_image(source_images, label_file.stem)
            if image_file is None:
                skipped_no_image += 1
                continue

            base_stem = label_file.stem
            candidate_stem = base_stem
            idx = 1
            while True:
                label_conflict = (target_labels / f"{candidate_stem}.txt").exists()
                image_conflict = any(
                    (target_images / f"{candidate_stem}{suffix}").exists()
                    for suffix in image_suffixes
                )
                if not label_conflict and not image_conflict:
                    break
                candidate_stem = f"{base_stem}__{source_tag}_{idx:03d}"
                idx += 1

            shutil.move(str(label_file), str(target_labels / f"{candidate_stem}.txt"))
            shutil.move(
                str(image_file), str(target_images / f"{candidate_stem}{image_file.suffix.lower()}")
            )
            moved_pairs += 1

        cleanup_dirs = [
            source_labels,
            source_labels.parent,
            source_images,
            source_images.parent,
            source_root,
        ]
        for cleanup_dir in cleanup_dirs:
            try:
                if cleanup_dir.exists() and cleanup_dir.is_dir():
                    cleanup_dir.rmdir()
            except OSError:
                pass

    return {
        "moved_pairs": moved_pairs,
        "skipped_no_image": skipped_no_image,
        "source_sets": len(source_sets),
    }


def discover_label_sources(labels_root: Path) -> List[Tuple[Path, Path, str]]:
    """
    Discover one or more manual-label sources under labels_root.
    Preferred layout:
    1) labels_root/images/train + labels_root/labels/train
    Legacy fallback:
    2) labels_root/<video_stem>/images/train + labels_root/<video_stem>/labels/train
    """
    sources: List[Tuple[Path, Path, str]] = []
    seen = set()

    direct_images = labels_root / "images" / "train"
    direct_labels = labels_root / "labels" / "train"
    if direct_images.exists() and direct_labels.exists():
        tag = _safe_tag(labels_root.name or "manual_labels")
        key = (str(direct_images.resolve()), str(direct_labels.resolve()))
        if key not in seen:
            sources.append((direct_images, direct_labels, tag))
            seen.add(key)

    for labels_dir in sorted(labels_root.rglob("labels/train")):
        source_root = labels_dir.parent.parent
        images_dir = source_root / "images" / "train"
        if not images_dir.exists():
            continue
        try:
            rel_root = source_root.relative_to(labels_root)
            tag_text = "__".join(rel_root.parts) if rel_root.parts else source_root.name
        except Exception:
            tag_text = source_root.name
        tag = _safe_tag(tag_text or "manual_labels")
        key = (str(images_dir.resolve()), str(labels_dir.resolve()))
        if key in seen:
            continue
        sources.append((images_dir, labels_dir, tag))
        seen.add(key)

    return sources


def build_dlc_labeled_dataset(
    labels_root: Path,
    project_path: Path,
    scorer: str,
    bodyparts: List[str],
    dataset_name: str,
):
    sources = discover_label_sources(labels_root)
    if len(sources) == 0:
        raise RuntimeError(
            f"No manual label sources found under {labels_root}. "
            "Expected labels/train + images/train (directly or per-video subfolders)."
        )

    out_dir = project_path / "labeled-data" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bodyparts_set = set(bodyparts)
    samples = []
    skipped_missing_image = 0
    skipped_empty = 0
    skipped_duplicate = 0
    used_rel_images = set()

    for images_dir, labels_dir, source_tag in sources:
        label_files = sorted(labels_dir.glob("*.txt"))
        for label_file in label_files:
            stem = label_file.stem
            image_file = find_matching_image(images_dir, stem)
            if image_file is None:
                skipped_missing_image += 1
                continue

            parsed = parse_manual_label_file(label_file)
            mapped_points: Dict[str, Tuple[float, float]] = {}
            for point_name, xy in parsed.items():
                mapped_name = map_point_name(point_name, bodyparts_set)
                if mapped_name is not None:
                    mapped_points[mapped_name] = xy

            if len(mapped_points) == 0:
                skipped_empty += 1
                continue

            dst_name = f"{source_tag}__{image_file.stem}{image_file.suffix.lower()}"
            rel_image = (Path("labeled-data") / dataset_name / dst_name).as_posix()
            if rel_image in used_rel_images:
                skipped_duplicate += 1
                continue
            used_rel_images.add(rel_image)

            dst_image = out_dir / dst_name
            if not dst_image.exists():
                shutil.copy2(image_file, dst_image)
            samples.append((rel_image, mapped_points))

    if len(samples) == 0:
        raise RuntimeError(
            "No valid labeled samples found. Check exported labels and bodypart names."
        )

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(index=[s[0] for s in samples], columns=columns, dtype=float)

    for rel_image, mapped_points in samples:
        for point_name, (x_val, y_val) in mapped_points.items():
            df.loc[rel_image, (scorer, point_name, "x")] = float(x_val)
            df.loc[rel_image, (scorer, point_name, "y")] = float(y_val)

    df.sort_index(inplace=True)
    csv_path = out_dir / f"CollectedData_{scorer}.csv"
    h5_path = out_dir / f"CollectedData_{scorer}.h5"
    df.to_csv(csv_path)
    try:
        df.to_hdf(h5_path, key="df_with_missing", mode="w")
    except Exception as exc:
        raise RuntimeError(
            "Failed to write H5 labels. Install pytables (`pip install tables`) and retry."
        ) from exc

    return {
        "dataset_dir": out_dir,
        "csv_path": csv_path,
        "h5_path": h5_path,
        "sample_count": len(samples),
        "source_count": len(sources),
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
    }


def ensure_dataset_in_config(
    cfg: dict,
    project_path: Path,
    dataset_name: str,
    video_path: Optional[Path],
) -> bool:
    """Register the labeled-data folder even when retraining without a loaded video."""
    video_sets = cfg.get("video_sets")
    if not isinstance(video_sets, dict):
        video_sets = {}
    video_key = str(
        video_path.expanduser().resolve()
        if video_path is not None
        else project_path / "videos" / f"{dataset_name}.mp4"
    )
    if video_key in video_sets:
        return False
    video_sets[video_key] = {}
    cfg["video_sets"] = video_sets
    return True


def find_exported_model_path(project_path: Path) -> Optional[Path]:
    exported_root = project_path / "exported-models"
    if not exported_root.exists():
        return None

    # Prefer a folder that directly contains pose_cfg + snapshots.
    pose_cfg_paths = sorted(
        exported_root.rglob("pose_cfg.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for pose_cfg in pose_cfg_paths:
        parent = pose_cfg.parent
        if any(parent.glob("snapshot*.index")):
            return parent
    if pose_cfg_paths:
        return pose_cfg_paths[0].parent
    return None


def find_snapshot_prefix(model_path: Path) -> Path:
    """Return a verified TensorFlow snapshot prefix from a trained model folder."""
    snapshots = sorted(
        model_path.rglob("snapshot*.index"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise RuntimeError(f"No trained snapshot*.index found under source model: {model_path}")

    prefix = snapshots[0].with_suffix("")
    if not list(prefix.parent.glob(prefix.name + ".data*")):
        raise RuntimeError(f"Snapshot data file is missing for: {prefix}")
    return prefix


def read_source_model_type(source_model_path: Path, source_snapshot: Path) -> str:
    """Read the network type from the selected trained model."""
    yaml = require_yaml()
    pose_configs = [source_snapshot.parent / "pose_cfg.yaml"]
    pose_configs.extend(sorted(source_model_path.rglob("pose_cfg.yaml")))
    source_pose_config = next((path for path in pose_configs if path.is_file()), None)
    if source_pose_config is None:
        raise RuntimeError(f"No pose_cfg.yaml found under source model: {source_model_path}")

    with open(source_pose_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    net_type = cfg.get("net_type")
    if not net_type:
        raise RuntimeError(f"Source model pose config has no net_type: {source_pose_config}")
    log(f"VERIFIED source pose config: {source_pose_config}")
    log(f"VERIFIED source network type: {net_type}")
    return str(net_type)


def verify_training_init_weights(project_path: Path, source_snapshot: Path):
    """Verify DLC generated the training config from the selected snapshot."""
    yaml = require_yaml()
    pose_configs = sorted(
        project_path.glob("dlc-models/iteration-*/**/train/pose_cfg.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not pose_configs:
        raise RuntimeError(f"DeepLabCut did not create a training pose_cfg.yaml under {project_path}")

    pose_config = pose_configs[0]
    with open(pose_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("init_weights") != str(source_snapshot):
        raise RuntimeError(
            "DeepLabCut did not initialize the training config from the selected snapshot"
        )
    log(f"VERIFIED fine-tune initialization: {pose_config}")
    log(f"VERIFIED source weights: {source_snapshot}")
    return pose_config


def create_training_dataset_from_snapshot(
    deeplabcut,
    config_path: Path,
    shuffle: int,
    source_snapshot: Path,
    source_net_type: str,
):
    """Create the DLC dataset without looking up generic pretrained weights."""
    training_module = importlib.import_module(
        "deeplabcut.generate_training_dataset.trainingsetmanipulation"
    )
    original_weight_lookup = training_module.auxfun_models.check_for_weights
    training_module.auxfun_models.check_for_weights = (
        lambda *_args, **_kwargs: str(source_snapshot)
    )
    try:
        deeplabcut.create_training_dataset(
            str(config_path),
            Shuffles=[shuffle],
            userfeedback=False,
            net_type=source_net_type,
        )
    finally:
        training_module.auxfun_models.check_for_weights = original_weight_lookup


def copy_exported_model(model_path: Path, output_path: Path) -> Path:
    if model_path.resolve() == output_path.resolve():
        raise RuntimeError("Source/exported model and retrained output folder must be different")
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"Retrained output folder must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_path, output_path, dirs_exist_ok=True)
    find_snapshot_prefix(output_path)
    log(f"VERIFIED retrained model output: {output_path}")
    return output_path


def run_dlc_retrain(
    config_path: Path,
    project_path: Path,
    source_model_path: Path,
    output_model_path: Path,
    shuffle: int,
    iterations: int,
):
    deeplabcut = require_deeplabcut()
    source_snapshot = find_snapshot_prefix(source_model_path)
    source_net_type = read_source_model_type(source_model_path, source_snapshot)
    log(f"VERIFIED trained source model: {source_model_path}")

    log("Creating/rebuilding training dataset from the selected trained snapshot ...")
    create_training_dataset_from_snapshot(
        deeplabcut,
        config_path,
        shuffle,
        source_snapshot,
        source_net_type,
    )

    verify_training_init_weights(project_path, source_snapshot)

    displayiters = max(100, min(1000, iterations // 20))
    saveiters = max(500, min(5000, iterations // 4))
    log(
        f"Training network (shuffle={shuffle}, maxiters={iterations}, "
        f"displayiters={displayiters}, saveiters={saveiters}) ..."
    )
    deeplabcut.train_network(
        str(config_path),
        shuffle=shuffle,
        maxiters=int(iterations),
        displayiters=int(displayiters),
        saveiters=int(saveiters),
    )

    log("Evaluating the trained network on the held-out test data ...")
    deeplabcut.evaluate_network(
        str(config_path),
        Shuffles=[shuffle],
        plotting=False,
    )
    log("VERIFIED train/test evaluation completed.")

    log("Exporting latest model ...")
    try:
        deeplabcut.export_model(str(config_path), shuffle=shuffle, make_tar=False)
    except TypeError:
        deeplabcut.export_model(str(config_path), shuffle=shuffle)
    except Exception as exc:
        log(f"Warning: export_model raised: {exc}")

    model_path = find_exported_model_path(project_path)
    if model_path is None:
        log("Warning: could not auto-detect exported model path.")
    else:
        log(f"Detected exported model path: {model_path}")
    if model_path is None:
        raise RuntimeError("Could not find the model exported by DeepLabCut")
    return copy_exported_model(model_path, output_model_path)


def rerun_single_video(
    run_model_script: Path,
    model_path: Path,
    cam_name: str,
    video_path: Path,
    calibration_dir: Optional[Path] = None,
    screen_start_x: Optional[float] = None,
    screen_pix_cm: Optional[float] = None,
    screen_y: Optional[float] = None,
):
    run_model_script = run_model_script.expanduser().resolve()
    if not run_model_script.exists():
        raise RuntimeError(f"run_model.py does not exist: {run_model_script}")

    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise RuntimeError(f"Video does not exist: {video_path}")

    arena_dir = run_model_script.parent
    cwd_before = Path.cwd()
    predictor = None
    try:
        os.chdir(arena_dir)
        if str(arena_dir) not in sys.path:
            sys.path.insert(0, str(arena_dir))

        spec = importlib.util.spec_from_file_location("preytouch_run_model", run_model_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed loading run_model module: {run_model_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.config.CALIBRATION_DIR = str(calibration_dir) if calibration_dir else module.config.CALIBRATION_DIR
        module.config.SCREEN_START_X_CM = screen_start_x
        module.config.SCREEN_PIX_CM = screen_pix_cm
        module.config.SCREEN_Y_CM = screen_y
        module.config.IS_SCREEN_CONFIGURED_FOR_POSE = screen_start_x is not None and screen_pix_cm is not None

        log(f"Running PreyTouch prediction on one video with model {model_path}: {video_path}")
        pose_module = importlib.import_module("analysis.pose")
        predictor = pose_module.DLCArenaPose(
            cam_name,
            model_path=str(model_path),
            is_use_db=False,
            is_raise_no_caliber=False,
        )
        predictor.predict_video(video_path=str(video_path))
        log("Prediction finished.")
    finally:
        try:
            if predictor is not None and hasattr(predictor, "close"):
                predictor.close()
        except Exception:
            pass
        os.chdir(cwd_before)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain DeepLabCut from manual_labels and optionally re-run PreyTouch prediction."
    )
    parser.add_argument("--dlc-config", required=True, help="Path to DLC project config.yaml")
    parser.add_argument(
        "--labels-root",
        required=True,
        help="Path to shared manual_labels root (images/train + labels/train).",
    )
    parser.add_argument("--video-path", default=None, help="Video path used for labels or prediction")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name under DLC labeled-data")
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle number")
    parser.add_argument("--iterations", type=int, default=5000, help="maxiters for train_network")
    parser.add_argument("--source-model", default=None, help="Required trained model used as initial weights")
    parser.add_argument("--output-model", default=None, help="Destination folder for the retrained model")
    parser.add_argument("--predict-only", action="store_true", help="Run prediction without retraining")
    parser.add_argument("--run-model-script", default=None, help="Path to PreyTouch Arena/run_model.py")
    parser.add_argument("--model-path", default=None, help="Optional explicit model folder path")
    parser.add_argument("--cam-name", default="top", help="Camera name for rerun")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--screen-start-x", type=float, default=None)
    parser.add_argument("--screen-pix-cm", type=float, default=None)
    parser.add_argument("--screen-y", type=float, default=None)
    args = parser.parse_args()

    dlc_config_path = Path(args.dlc_config).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve()
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None
    run_model_script = (
        Path(args.run_model_script).expanduser().resolve() if args.run_model_script else None
    )
    model_path_override = Path(args.model_path).expanduser().resolve() if args.model_path else None
    source_model_path = Path(args.source_model).expanduser().resolve() if args.source_model else None
    output_model_path = Path(args.output_model).expanduser().resolve() if args.output_model else None

    if args.predict_only:
        if not run_model_script or not video_path or not model_path_override:
            raise RuntimeError("Prediction requires run-model-script, video-path, and model-path")
        find_snapshot_prefix(model_path_override)
        rerun_single_video(
            run_model_script, model_path_override, args.cam_name, video_path,
            Path(args.calibration_dir).expanduser().resolve() if args.calibration_dir else None,
            args.screen_start_x, args.screen_pix_cm, args.screen_y,
        )
        log("Prediction done.")
        return

    if not dlc_config_path.exists():
        raise RuntimeError(f"DLC config not found: {dlc_config_path}")
    if not labels_root.exists():
        raise RuntimeError(f"labels-root not found: {labels_root}")
    if model_path_override is not None and not model_path_override.exists():
        raise RuntimeError(f"model-path not found: {model_path_override}")
    if source_model_path is None or not source_model_path.exists():
        raise RuntimeError("A valid --source-model is required; retraining from scratch is not allowed")
    if output_model_path is None:
        raise RuntimeError("--output-model is required")

    flatten_stats = flatten_manual_labels_root(labels_root)
    log(f"DLC config: {dlc_config_path}")
    log(f"Labels root: {labels_root}")
    if flatten_stats["source_sets"] > 0:
        log(
            "Flattened manual labels root: "
            f"sources={flatten_stats['source_sets']}, "
            f"moved_pairs={flatten_stats['moved_pairs']}, "
            f"skipped_no_image={flatten_stats['skipped_no_image']}"
        )
    if video_path:
        log(f"Video path: {video_path}")

    cfg, project_path, scorer, bodyparts = load_dlc_project_config(dlc_config_path)
    cfg["date"] = str(cfg.get("date", ""))
    cfg["TrainingFraction"] = [0.8]
    cfg["engine"] = "tensorflow"
    cfg["multianimalproject"] = False
    cfg.setdefault("pcutoff", 0.4)
    cfg.setdefault("colormap", "jet")
    if not cfg.get("project_path"):
        project_path = output_model_path.parent / f"{output_model_path.name}_dlc_project"
        project_path.mkdir(parents=True, exist_ok=True)
        for generated_dir in ("dlc-models", "exported-models"):
            shutil.rmtree(project_path / generated_dir, ignore_errors=True)
        cfg["project_path"] = str(project_path)
        dlc_config_path = project_path / "config.yaml"
        log(f"Created DLC working project from template: {dlc_config_path}")
        log("Reset generated model state so training starts from the selected source snapshot.")
    save_dlc_project_config(dlc_config_path, cfg)
    log("VERIFIED training/test split: 80%/20%")
    log("VERIFIED training engine: tensorflow (required by snapshot*.index source weights)")
    dataset_name = args.dataset_name or (video_path.stem if video_path else labels_root.name)
    log(f"DLC project path: {project_path}")
    log(f"Scorer: {scorer}")
    log(f"Bodyparts ({len(bodyparts)}): {bodyparts}")
    log(f"Dataset name: {dataset_name}")

    prep_stats = build_dlc_labeled_dataset(
        labels_root=labels_root,
        project_path=project_path,
        scorer=scorer,
        bodyparts=bodyparts,
        dataset_name=dataset_name,
    )
    if prep_stats["sample_count"] < 2:
        raise RuntimeError("At least two valid manually labeled images are required for a train/test split")
    log(
        "Prepared labeled-data dataset: "
        f"sources={prep_stats['source_count']}, "
        f"samples={prep_stats['sample_count']}, "
        f"skipped_missing_image={prep_stats['skipped_missing_image']}, "
        f"skipped_empty={prep_stats['skipped_empty']}, "
        f"skipped_duplicate={prep_stats['skipped_duplicate']}"
    )
    log(f"Collected CSV: {prep_stats['csv_path']}")
    log(f"Collected H5: {prep_stats['h5_path']}")
    log(f"VERIFIED new manual training samples: {prep_stats['sample_count']}")

    if ensure_dataset_in_config(cfg, project_path, dataset_name, video_path):
        save_dlc_project_config(dlc_config_path, cfg)
        log(f"Registered labeled-data dataset in DLC config: {dataset_name}")

    model_path = run_dlc_retrain(
        config_path=dlc_config_path,
        project_path=project_path,
        source_model_path=source_model_path,
        output_model_path=output_model_path,
        shuffle=int(args.shuffle),
        iterations=int(args.iterations),
    )

    log(f"Retraining done. Retrained model: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        log(f"ERROR: {exc}")
        sys.exit(1)
