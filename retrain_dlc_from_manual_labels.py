#!/usr/bin/env python3
"""
Retrain DeepLabCut from manual_labels exports and optionally rerun PreyTouch on one video.
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
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
    except Exception as exc:
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
    project_path = Path(cfg.get("project_path", config_path.parent)).expanduser().resolve()
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


def validate_image_readable(image_path: Path) -> Optional[str]:
    """
    Best-effort validation that an image can be fully decoded.

    DeepLabCut can hang when the training loader thread dies on a corrupt/truncated image.
    Catch this early and fail fast with a helpful error.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        # If Pillow is not available, skip validation (DLC envs typically include it).
        return None

    try:
        with Image.open(image_path) as im:
            im.load()  # force full decode
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

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


def prune_labels_without_images(labels_root: Path) -> Dict[str, int]:
    """Delete manual label files that do not have a matching image file."""
    sources = discover_label_sources(labels_root)
    checked = 0
    removed = 0
    remove_errors = 0

    for images_dir, labels_dir, _source_tag in sources:
        for label_file in sorted(labels_dir.glob("*.txt")):
            checked += 1
            image_file = find_matching_image(images_dir, label_file.stem)
            if image_file is not None:
                continue
            try:
                label_file.unlink()
                removed += 1
            except OSError:
                remove_errors += 1

    return {
        "source_count": len(sources),
        "checked_labels": checked,
        "removed_labels": removed,
        "remove_errors": remove_errors,
    }


def prune_bad_image_pairs(labels_root: Path) -> Dict[str, int]:
    """
    Move unreadable image+label pairs out of the active manual_labels tree.

    We keep the files under labels_root/bckup/pruned_bad_pairs so the user can
    inspect or recover them later, but DLC will no longer see them.
    """
    sources = discover_label_sources(labels_root)
    checked = 0
    bad_pairs = 0
    moved_labels = 0
    moved_images = 0
    move_errors = 0

    backup_root = labels_root / "bckup" / "pruned_bad_pairs"

    for images_dir, labels_dir, source_tag in sources:
        backup_images_dir = backup_root / source_tag / "images" / "train"
        backup_labels_dir = backup_root / source_tag / "labels" / "train"

        for label_file in sorted(labels_dir.glob("*.txt")):
            image_file = find_matching_image(images_dir, label_file.stem)
            if image_file is None:
                continue

            checked += 1
            decode_error = validate_image_readable(image_file)
            if not decode_error:
                continue

            bad_pairs += 1
            try:
                backup_images_dir.mkdir(parents=True, exist_ok=True)
                backup_labels_dir.mkdir(parents=True, exist_ok=True)

                base_stem = label_file.stem
                suffix_idx = 0
                while True:
                    candidate_stem = (
                        base_stem if suffix_idx == 0 else f"{base_stem}__dup_{suffix_idx:03d}"
                    )
                    label_dst = backup_labels_dir / f"{candidate_stem}{label_file.suffix}"
                    image_dst = backup_images_dir / f"{candidate_stem}{image_file.suffix.lower()}"
                    if not label_dst.exists() and not image_dst.exists():
                        break
                    suffix_idx += 1

                shutil.move(str(label_file), str(label_dst))
                moved_labels += 1

                shutil.move(str(image_file), str(image_dst))
                moved_images += 1
            except OSError:
                move_errors += 1

    return {
        "source_count": len(sources),
        "checked_images": checked,
        "bad_pairs": bad_pairs,
        "moved_labels": moved_labels,
        "moved_images": moved_images,
        "move_errors": move_errors,
        "backup_root": str(backup_root),
    }


def build_dlc_labeled_dataset(
    labels_root: Path,
    project_path: Path,
    scorer: str,
    bodyparts: List[str],
    dataset_name: str,
    *,
    skip_bad_images: bool = False,
):
    sources = discover_label_sources(labels_root)
    if len(sources) == 0:
        raise RuntimeError(
            f"No manual label sources found under {labels_root}. "
            "Expected labels/train + images/train (directly or per-video subfolders)."
        )

    out_dir = project_path / "labeled-data" / dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bodyparts_set = set(bodyparts)
    samples = []
    skipped_missing_image = 0
    skipped_bad_image = 0
    skipped_empty = 0
    skipped_duplicate = 0
    used_rel_images = set()
    bad_images: List[Tuple[Path, str]] = []

    for images_dir, labels_dir, source_tag in sources:
        label_files = sorted(labels_dir.glob("*.txt"))
        for label_file in label_files:
            stem = label_file.stem
            image_file = find_matching_image(images_dir, stem)
            if image_file is None:
                skipped_missing_image += 1
                continue

            decode_error = validate_image_readable(image_file)
            if decode_error:
                if skip_bad_images:
                    skipped_bad_image += 1
                    continue
                bad_images.append((image_file, decode_error))
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
            shutil.copy2(image_file, dst_image)
            samples.append((rel_image, mapped_points))

    if bad_images:
        lines = "\n".join(
            f"  - {p} ({err})" for p, err in bad_images[:10]
        )
        more = "" if len(bad_images) <= 10 else f"\n  ... and {len(bad_images) - 10} more"
        raise RuntimeError(
            "Found unreadable/corrupt images referenced by your manual labels. "
            "Fix/re-export or delete these files and rerun.\n"
            f"{lines}{more}"
        )

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
        "skipped_bad_image": skipped_bad_image,
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
    }


def ensure_video_in_config(cfg: dict, video_path: Optional[Path]) -> bool:
    if video_path is None:
        return False
    video_sets = cfg.get("video_sets")
    if not isinstance(video_sets, dict):
        video_sets = {}
    video_key = str(video_path.expanduser().resolve())
    if video_key in video_sets:
        return False
    video_sets[video_key] = {}
    cfg["video_sets"] = video_sets
    return True


def ensure_dataset_in_config(cfg: dict, project_path: Path, dataset_name: str) -> bool:
    """
    DeepLabCut's create_training_dataset discovers labeled-data folders by splitting
    the paths in config['video_sets'] into stems. Our manual labels live under:

        labeled-data/<dataset_name>/CollectedData_<scorer>.{csv,h5}

    If <dataset_name> is not represented in video_sets, DLC may ignore this folder.
    To force inclusion (without requiring a real video path), we add a dummy entry
    with stem == dataset_name (and touch an empty file so existence checks won't fail).
    """
    video_sets = cfg.get("video_sets")
    if not isinstance(video_sets, dict):
        video_sets = {}

    dummy_video = (project_path / f"{dataset_name}.mp4").expanduser().resolve()
    try:
        dummy_video.parent.mkdir(parents=True, exist_ok=True)
        dummy_video.touch(exist_ok=True)
    except Exception:
        # Still add the key; most DLC code only needs the stem.
        pass

    video_key = str(dummy_video)
    if video_key in video_sets:
        cfg["video_sets"] = video_sets
        return False

    video_sets[video_key] = {}
    cfg["video_sets"] = video_sets
    return True


def find_latest_documentation_pickle(project_path: Path, iteration: int, shuffle: int) -> Optional[Path]:
    root = project_path / "training-datasets" / f"iteration-{iteration}"
    if not root.exists():
        return None
    candidates = list(root.rglob(f"Documentation_data-*shuffle{shuffle}.pickle"))
    if not candidates:
        candidates = list(root.rglob("Documentation_data-*.pickle"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def count_images_from_labeled_data(doc_pickle: Path, dataset_name: str) -> Optional[int]:
    try:
        import pickle
        obj = pickle.loads(doc_pickle.read_bytes())
    except Exception:
        return None

    # Typical DLC format (as seen in this repo): [images, trainIndices, testIndices, trainFraction]
    images = None
    if isinstance(obj, list) and len(obj) >= 1:
        images = obj[0]
    if images is None:
        return None

    needle = f"('labeled-data', '{dataset_name}',"
    count = 0
    for item in images:
        # item is often a dict, but can be stringified depending on DLC version
        s = str(item)
        if needle in s:
            count += 1
    return count


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


def _patch_yaml_scalar_line(path: Path, key: str, value: str):
    lines = path.read_text(encoding="utf-8").splitlines(True)
    out = []
    replaced = False
    for ln in lines:
        if ln.lstrip().startswith(f"{key}:"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}{key}: {value}\n")
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(f"\n{key}: {value}\n")
    path.write_text("".join(out), encoding="utf-8")


def _find_pose_cfg_paths(
    project_path: Path, iteration: int, shuffle: int
) -> Tuple[Optional[Path], Optional[Path]]:
    iter_dir = project_path / "dlc-models" / f"iteration-{iteration}"
    if not iter_dir.exists():
        return None, None

    candidates: List[Tuple[Path, Optional[Path]]] = []
    for model_dir in sorted(iter_dir.glob(f"*shuffle{shuffle}")):
        if not model_dir.is_dir():
            continue
        train_pose = model_dir / "train" / "pose_cfg.yaml"
        if not train_pose.exists():
            continue
        test_pose = model_dir / "test" / "pose_cfg.yaml"
        candidates.append((train_pose, test_pose if test_pose.exists() else None))

    if not candidates:
        return None, None

    candidates.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
    return candidates[0]


def run_dlc_retrain(
    config_path: Path,
    project_path: Path,
    iteration: int,
    shuffle: int,
    iterations: int,
    dataset_name: str,
    *,
    net_type: Optional[str] = None,
    augmenter_type: Optional[str] = None,
    posecfg_template: Optional[Path] = None,
    init_weights: Optional[Path] = None,
    prepare_only: bool = False,
):
    deeplabcut = require_deeplabcut()

    log("Creating/rebuilding training dataset ...")
    try:
        kwargs = {"Shuffles": [shuffle], "userfeedback": False}
        if net_type:
            kwargs["net_type"] = str(net_type)
        if augmenter_type:
            kwargs["augmenter_type"] = str(augmenter_type)
        if posecfg_template:
            kwargs["posecfg_template"] = str(posecfg_template)
        deeplabcut.create_training_dataset(str(config_path), **kwargs)
    except TypeError:
        deeplabcut.create_training_dataset(str(config_path), Shuffles=[shuffle])
    except Exception as exc:
        log(f"Warning: create_training_dataset raised: {exc}")
        log("Continuing with train_network (existing dataset may still be usable).")

    doc_pickle = find_latest_documentation_pickle(project_path, iteration=iteration, shuffle=shuffle)
    if doc_pickle is not None:
        cnt = count_images_from_labeled_data(doc_pickle, dataset_name=dataset_name)
        if cnt is not None:
            log(f"Training dataset includes {cnt} images from labeled-data/{dataset_name}")

    if prepare_only:
        log("prepare_only=1: skipping train_network/export_model.")
        return None

    if init_weights is not None:
        train_pose_cfg, test_pose_cfg = _find_pose_cfg_paths(
            project_path=project_path, iteration=iteration, shuffle=shuffle
        )
        if train_pose_cfg is None:
            log("Warning: could not find train/pose_cfg.yaml to patch init_weights.")
        else:
            _patch_yaml_scalar_line(train_pose_cfg, "init_weights", init_weights.as_posix())
            if test_pose_cfg is not None:
                _patch_yaml_scalar_line(test_pose_cfg, "init_weights", init_weights.as_posix())
            log(f"Patched init_weights -> {init_weights} in pose_cfg.yaml")

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
    return model_path


def update_preytouch_predict_config(run_model_script: Path, model_name: str, model_path: Path):
    arena_dir = run_model_script.expanduser().resolve().parent
    predict_cfg_path = arena_dir / "configurations" / "predict_config.json"
    if not predict_cfg_path.exists():
        raise RuntimeError(f"Predict config not found: {predict_cfg_path}")

    with open(predict_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if model_name not in cfg:
        raise RuntimeError(
            f"Model '{model_name}' is not in {predict_cfg_path}. Add it first in PreyTouch."
        )

    backup_path = predict_cfg_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(predict_cfg_path, backup_path)

    old_model_path = cfg[model_name].get("model_path")
    cfg[model_name]["model_path"] = str(model_path)
    with open(predict_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log(
        f"Updated predict config model_path for '{model_name}': "
        f"{old_model_path} -> {model_path}"
    )
    return predict_cfg_path


def rerun_single_video(run_model_script: Path, model_name: str, cam_name: str, video_path: Path):
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

        pred_conf = module.config.load_configuration("predict")
        if model_name not in pred_conf:
            raise RuntimeError(f"Model '{model_name}' not found in PreyTouch predict config.")

        log(f"Running PreyTouch prediction on one video: {video_path}")
        predictor = module.load_predictor(pred_conf, model_name, cam_name)
        module.predict_video(predictor, str(video_path))
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
    parser.add_argument("--video-path", default=None, help="Video path used for labels and optional rerun")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name under DLC labeled-data")
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle number")
    parser.add_argument("--iterations", type=int, default=5000, help="maxiters for train_network")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only build labeled-data + create training dataset; skip training/export.",
    )
    parser.add_argument(
        "--skip-bad-images",
        action="store_true",
        help="Skip manual-label samples whose image files are corrupt/unreadable instead of failing fast.",
    )
    parser.add_argument(
        "--prune-missing-image-labels",
        action="store_true",
        help="Delete label txt files that do not have a matching image file before dataset build.",
    )
    parser.add_argument(
        "--prune-bad-images",
        action="store_true",
        help="Move unreadable image+label pairs out of manual_labels before dataset build.",
    )
    parser.add_argument(
        "--net-type",
        default=None,
        help="Optional DLC net_type passed to create_training_dataset (e.g. resnet_152).",
    )
    parser.add_argument(
        "--augmenter-type",
        default=None,
        help="Optional DLC augmenter_type passed to create_training_dataset (e.g. imgaug).",
    )
    parser.add_argument(
        "--posecfg-template",
        default=None,
        help="Optional path to a pose_cfg.yaml template for create_training_dataset.",
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        help="Optional checkpoint base path to fine-tune from (e.g. /path/to/snapshot-500000).",
    )
    parser.add_argument("--run-model-script", default=None, help="Path to PreyTouch Arena/run_model.py")
    parser.add_argument("--model-name", default=None, help="Model key from PreyTouch predict_config.json")
    parser.add_argument("--model-path", default=None, help="Optional explicit model folder path")
    parser.add_argument("--cam-name", default="top", help="Camera name for rerun")
    args = parser.parse_args()

    dlc_config_path = Path(args.dlc_config).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve()
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None
    run_model_script = (
        Path(args.run_model_script).expanduser().resolve() if args.run_model_script else None
    )
    model_path_override = Path(args.model_path).expanduser().resolve() if args.model_path else None
    posecfg_template = (
        Path(args.posecfg_template).expanduser().resolve() if args.posecfg_template else None
    )
    init_weights = Path(args.init_weights).expanduser().resolve() if args.init_weights else None

    if not dlc_config_path.exists():
        raise RuntimeError(f"DLC config not found: {dlc_config_path}")
    if not labels_root.exists():
        raise RuntimeError(f"labels-root not found: {labels_root}")
    if model_path_override is not None and not model_path_override.exists():
        raise RuntimeError(f"model-path not found: {model_path_override}")
    if posecfg_template is not None and not posecfg_template.exists():
        raise RuntimeError(f"posecfg-template not found: {posecfg_template}")

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
    if args.prune_missing_image_labels:
        prune_stats = prune_labels_without_images(labels_root)
        log(
            "Pruned labels without images: "
            f"sources={prune_stats['source_count']}, "
            f"checked={prune_stats['checked_labels']}, "
            f"removed={prune_stats['removed_labels']}, "
            f"remove_errors={prune_stats['remove_errors']}"
        )
    if args.prune_bad_images:
        prune_bad_stats = prune_bad_image_pairs(labels_root)
        log(
            "Pruned bad image pairs: "
            f"sources={prune_bad_stats['source_count']}, "
            f"checked={prune_bad_stats['checked_images']}, "
            f"bad_pairs={prune_bad_stats['bad_pairs']}, "
            f"moved_labels={prune_bad_stats['moved_labels']}, "
            f"moved_images={prune_bad_stats['moved_images']}, "
            f"move_errors={prune_bad_stats['move_errors']}, "
            f"backup_root={prune_bad_stats['backup_root']}"
        )
    if video_path:
        log(f"Video path: {video_path}")

    cfg, project_path, scorer, bodyparts = load_dlc_project_config(dlc_config_path)
    iteration = int(cfg.get("iteration", 0))
    dataset_name = args.dataset_name or labels_root.name
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
        skip_bad_images=bool(args.skip_bad_images),
    )
    log(
        "Prepared labeled-data dataset: "
        f"sources={prep_stats['source_count']}, "
        f"samples={prep_stats['sample_count']}, "
        f"skipped_missing_image={prep_stats['skipped_missing_image']}, "
        f"skipped_bad_image={prep_stats['skipped_bad_image']}, "
        f"skipped_empty={prep_stats['skipped_empty']}, "
        f"skipped_duplicate={prep_stats['skipped_duplicate']}"
    )
    log(f"Collected CSV: {prep_stats['csv_path']}")
    log(f"Collected H5: {prep_stats['h5_path']}")

    # Ensure DLC will include this dataset in create_training_dataset even if user didn't provide a real --video-path
    changed_cfg = False
    if ensure_dataset_in_config(cfg, project_path=project_path, dataset_name=dataset_name):
        changed_cfg = True
        log(f"Updated DLC config video_sets with dummy dataset video: {dataset_name}.mp4")

    if ensure_video_in_config(cfg, video_path):
        changed_cfg = True
        save_dlc_project_config(dlc_config_path, cfg)
        log("Updated DLC config video_sets with current video.")
    elif changed_cfg:
        save_dlc_project_config(dlc_config_path, cfg)

    model_path = run_dlc_retrain(
        config_path=dlc_config_path,
        project_path=project_path,
        iteration=iteration,
        shuffle=int(args.shuffle),
        iterations=int(args.iterations),
        dataset_name=dataset_name,
        net_type=args.net_type,
        augmenter_type=args.augmenter_type,
        posecfg_template=posecfg_template,
        init_weights=init_weights,
        prepare_only=bool(args.prepare_only),
    )

    if args.prepare_only:
        log("Done (prepare_only).")
        return

    if run_model_script and args.model_name and video_path:
        effective_model_path = model_path_override if model_path_override is not None else model_path
        if effective_model_path is not None:
            update_preytouch_predict_config(run_model_script, args.model_name, effective_model_path)
        else:
            log("No exported model path detected; rerun will use current model_path in predict config.")
        rerun_single_video(
            run_model_script=run_model_script,
            model_name=args.model_name,
            cam_name=args.cam_name,
            video_path=video_path,
        )
    else:
        log(
            "Skipping PreyTouch rerun. To enable rerun, provide --run-model-script, "
            "--model-name, and --video-path."
        )

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
