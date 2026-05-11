#!/usr/bin/env python3
"""
Reapply Arena camera calibration to an already-edited pose file.

This script expects a flat pose table with columns like:
  nose_cam_x, nose_cam_y, nose_x, nose_y
and rewrites the calibrated *_x / *_y columns from the edited *_cam_x / *_cam_y.
"""

import argparse
import importlib.util
from importlib.machinery import ModuleSpec
import os
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CALIBRATION_DIR = Path("/Volumes/Data/Bareket/arenas_configs/zeology low/calibrations")


def log(message: str):
    print(message, flush=True)


def guess_default_arena_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.home() / "Dev" / "PreyTouch" / "Arena",
        here.parents[2] / "Arena",
        here.parent / "Arena",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def guess_default_calibration_dir(arena_dir: Optional[Path] = None) -> Optional[Path]:
    candidates = [DEFAULT_CALIBRATION_DIR]
    if arena_dir is not None:
        candidates.append(arena_dir.resolve().parent / "output" / "calibrations")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate.resolve()
    return None


def infer_cam_and_date(
    pose_path: Path,
    video_path: Optional[Path],
    cam_name: Optional[str],
    image_date: Optional[str],
) -> Tuple[str, str]:
    if cam_name and image_date:
        return cam_name, image_date

    stems = []
    if video_path is not None:
        stems.append(video_path.stem)
    stems.append(pose_path.stem.split("__")[-1])
    stems.append(pose_path.stem)

    for stem in stems:
        match = re.match(r"(?P<cam>[^_]+)_(?P<date>\d{8}T\d{6})", stem)
        if match:
            inferred_cam = cam_name or match.group("cam")
            inferred_date = image_date or match.group("date")
            return inferred_cam, inferred_date

    raise RuntimeError(
        "Could not infer cam/date. Provide --cam-name and --image-date explicitly."
    )


def load_pose_table(pose_path: Path) -> pd.DataFrame:
    if pose_path.suffix.lower() == ".parquet":
        return pd.read_parquet(pose_path)
    if pose_path.suffix.lower() == ".csv":
        return pd.read_csv(pose_path)
    raise RuntimeError(f"Unsupported pose file type: {pose_path.suffix}")


def save_pose_table(df: pd.DataFrame, output_path: Path):
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path)
        return
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    raise RuntimeError(f"Unsupported output file type: {output_path.suffix}")


def discover_bodyparts(df: pd.DataFrame) -> List[str]:
    bodyparts = []
    for col in df.columns:
        col_name = str(col)
        if not col_name.endswith("_cam_x"):
            continue
        bodypart = col_name[:-6]
        if f"{bodypart}_cam_y" in df.columns:
            bodyparts.append(bodypart)
    return sorted(set(bodyparts))


def load_charuco_estimator(
    arena_dir: Path,
    cam_name: str,
    image_date: str,
    calibration_dir: Optional[Path],
):
    _install_optional_arena_import_stubs()
    cwd_before = Path.cwd()
    os.chdir(arena_dir)
    try:
        if str(arena_dir) not in sys.path:
            sys.path.insert(0, str(arena_dir))

        import config  # type: ignore
        from calibration import CharucoEstimator  # type: ignore

        if calibration_dir is not None:
            config.CALIBRATION_DIR = calibration_dir.as_posix()

        estimator = CharucoEstimator(cam_name, is_debug=False)
        estimator.set_image_date_and_load(image_date)
        estimator.state = 2
        return estimator
    finally:
        os.chdir(cwd_before)


class _NoOpAxis:
    def imshow(self, *args, **kwargs):
        return None

    def axis(self, *args, **kwargs):
        return None


class _NoOpFigure:
    def tight_layout(self):
        return None

    def savefig(self, *args, **kwargs):
        return None


def _module_available(module_name: str) -> bool:
    """Check import availability without crashing on frozen-app modules with missing __spec__."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ValueError, ImportError, AttributeError):
        return False


def _install_optional_arena_import_stubs():
    """Provide lightweight stubs for Arena imports that calibration does not actually use."""
    if not _module_available("matplotlib"):
        matplotlib_mod = types.ModuleType("matplotlib")
        pyplot_mod = types.ModuleType("matplotlib.pyplot")
        matplotlib_mod.__spec__ = ModuleSpec("matplotlib", loader=None, is_package=True)
        matplotlib_mod.__path__ = []
        pyplot_mod.__spec__ = ModuleSpec("matplotlib.pyplot", loader=None)

        def _subplots(rows=1, cols=1, **kwargs):
            fig = _NoOpFigure()
            if rows == 1 and cols == 1:
                axes = _NoOpAxis()
            else:
                axes = np.empty((rows, cols), dtype=object)
                for row_idx in range(rows):
                    for col_idx in range(cols):
                        axes[row_idx, col_idx] = _NoOpAxis()
            return fig, axes

        pyplot_mod.subplots = _subplots
        pyplot_mod.close = lambda *args, **kwargs: None
        pyplot_mod.imshow = lambda *args, **kwargs: None
        pyplot_mod.show = lambda *args, **kwargs: None
        matplotlib_mod.pyplot = pyplot_mod
        sys.modules.setdefault("matplotlib", matplotlib_mod)
        sys.modules.setdefault("matplotlib.pyplot", pyplot_mod)

    if not _module_available("filterpy"):
        filterpy_mod = types.ModuleType("filterpy")
        kalman_mod = types.ModuleType("filterpy.kalman")
        filterpy_mod.__spec__ = ModuleSpec("filterpy", loader=None, is_package=True)
        filterpy_mod.__path__ = []
        kalman_mod.__spec__ = ModuleSpec("filterpy.kalman", loader=None)

        class KalmanFilter:  # pragma: no cover - only used when Arena drags in filterpy unexpectedly.
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        kalman_mod.KalmanFilter = KalmanFilter
        filterpy_mod.kalman = kalman_mod
        sys.modules.setdefault("filterpy", filterpy_mod)
        sys.modules.setdefault("filterpy.kalman", kalman_mod)


def reapply_calibration(df: pd.DataFrame, estimator, bodyparts: List[str]) -> Tuple[pd.DataFrame, int]:
    updated_cells = 0

    for bodypart in bodyparts:
        cam_x_col = f"{bodypart}_cam_x"
        cam_y_col = f"{bodypart}_cam_y"
        x_col = f"{bodypart}_x"
        y_col = f"{bodypart}_y"

        if x_col not in df.columns:
            df[x_col] = np.nan
        if y_col not in df.columns:
            df[y_col] = np.nan

        valid_mask = df[cam_x_col].notna() & df[cam_y_col].notna()
        df.loc[~valid_mask, [x_col, y_col]] = np.nan
        if not valid_mask.any():
            log(f"{bodypart}: 0 rows with valid cam coordinates")
            continue

        xs = []
        ys = []
        valid_rows = df.index[valid_mask]
        for row_idx in valid_rows:
            cam_x = float(df.at[row_idx, cam_x_col])
            cam_y = float(df.at[row_idx, cam_y_col])
            try:
                x_val, y_val = estimator.get_location(cam_x, cam_y)
            except Exception:
                x_val, y_val = np.nan, np.nan
            xs.append(x_val)
            ys.append(y_val)

        df.loc[valid_rows, x_col] = xs
        df.loc[valid_rows, y_col] = ys
        updated_cells += 2 * len(valid_rows)
        log(f"{bodypart}: recalibrated {len(valid_rows)} rows")

    return df, updated_cells


def run_reapply_calibration(
    pose_file: str,
    video_path: Optional[str] = None,
    arena_dir: Optional[str] = None,
    calibration_dir: Optional[str] = None,
    cam_name: Optional[str] = None,
    image_date: Optional[str] = None,
    output_path: Optional[str] = None,
    keep_backup: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run calibration reapplication and return a small execution summary."""
    log_fn = logger or log

    pose_path = Path(pose_file).expanduser().resolve()
    video_path_obj = Path(video_path).expanduser().resolve() if video_path else None
    arena_dir_obj = (
        Path(arena_dir).expanduser().resolve()
        if arena_dir else guess_default_arena_dir().resolve()
    )
    calibration_dir_obj = (
        Path(calibration_dir).expanduser().resolve()
        if calibration_dir else guess_default_calibration_dir(arena_dir_obj)
    )
    output_path_obj = (
        Path(output_path).expanduser().resolve()
        if output_path else pose_path
    )

    if not pose_path.exists():
        raise RuntimeError(f"Pose file not found: {pose_path}")
    if video_path_obj is not None and not video_path_obj.exists():
        raise RuntimeError(f"Video path not found: {video_path_obj}")
    if not arena_dir_obj.exists():
        raise RuntimeError(f"Arena dir not found: {arena_dir_obj}")
    if calibration_dir_obj is not None and not calibration_dir_obj.exists():
        raise RuntimeError(f"Calibration dir not found: {calibration_dir_obj}")

    resolved_cam_name, resolved_image_date = infer_cam_and_date(
        pose_path=pose_path,
        video_path=video_path_obj,
        cam_name=cam_name,
        image_date=image_date,
    )

    log_fn(f"Pose file: {pose_path}")
    log_fn(f"Arena dir: {arena_dir_obj}")
    if calibration_dir_obj is not None:
        log_fn(f"Calibration dir: {calibration_dir_obj}")
    else:
        log_fn("Calibration dir: Arena config default")
    log_fn(f"Camera: {resolved_cam_name}")
    log_fn(f"Calibration date: {resolved_image_date}")

    df = load_pose_table(pose_path)
    bodyparts = discover_bodyparts(df)
    if not bodyparts:
        raise RuntimeError("No *_cam_x / *_cam_y bodypart columns found.")
    log_fn(f"Bodyparts with cam coordinates: {bodyparts}")

    estimator = load_charuco_estimator(
        arena_dir=arena_dir_obj,
        cam_name=resolved_cam_name,
        image_date=resolved_image_date,
        calibration_dir=calibration_dir_obj,
    )
    df, updated_cells = reapply_calibration(df, estimator, bodyparts)

    backup_path = None
    if output_path_obj == pose_path:
        backup_path = pose_path.with_suffix(pose_path.suffix + ".pre_recalibration.bak")
        if keep_backup and not backup_path.exists():
            shutil.copy2(pose_path, backup_path)
            log_fn(f"Backup created: {backup_path}")

    tmp_output = output_path_obj.with_name(f"{output_path_obj.name}.tmp{output_path_obj.suffix}")
    save_pose_table(df, tmp_output)
    tmp_output.replace(output_path_obj)
    if backup_path is not None and not keep_backup and backup_path.exists():
        backup_path.unlink()
        log_fn(f"Removed backup: {backup_path}")
    log_fn(f"Saved recalibrated pose file: {output_path_obj}")
    log_fn(f"Updated calibrated coordinate cells: {updated_cells}")

    return {
        "pose_path": str(pose_path),
        "video_path": str(video_path_obj) if video_path_obj is not None else None,
        "output_path": str(output_path_obj),
        "arena_dir": str(arena_dir_obj),
        "calibration_dir": str(calibration_dir_obj) if calibration_dir_obj is not None else None,
        "cam_name": resolved_cam_name,
        "image_date": resolved_image_date,
        "bodyparts": bodyparts,
        "updated_cells": int(updated_cells),
        "kept_backup": bool(keep_backup),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reapply Arena calibration to an edited pose parquet/csv."
    )
    parser.add_argument("--pose-file", required=True, help="Edited pose parquet/csv")
    parser.add_argument("--video-path", default=None, help="Matching video path (optional but recommended)")
    parser.add_argument("--arena-dir", default=None, help="Path to PreyTouch Arena directory")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help=(
            "Override calibration directory "
            f"(default: {DEFAULT_CALIBRATION_DIR} if available)"
        ),
    )
    parser.add_argument("--cam-name", default=None, help="Camera name, e.g. front or top")
    parser.add_argument("--image-date", default=None, help="Calibration date key, e.g. 20260221T110059")
    parser.add_argument("--output-path", default=None, help="Output parquet/csv path (default: overwrite input)")
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep/create a .pre_recalibration.bak file when overwriting the input pose file",
    )
    args = parser.parse_args()

    run_reapply_calibration(
        pose_file=args.pose_file,
        video_path=args.video_path,
        arena_dir=args.arena_dir,
        calibration_dir=args.calibration_dir,
        cam_name=args.cam_name,
        image_date=args.image_date,
        output_path=args.output_path,
        keep_backup=bool(args.keep_backup),
        logger=log,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
