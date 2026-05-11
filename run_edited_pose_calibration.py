#!/usr/bin/env python3
"""Thin wrapper for recalibrating the currently edited pose file."""

import argparse
import sys

from reapply_pose_calibration import DEFAULT_CALIBRATION_DIR, run_reapply_calibration

DEFAULT_APP_CALIBRATION_DIR = str(DEFAULT_CALIBRATION_DIR)


def run_edited_pose_calibration_job(
    pose_file: str,
    video_path: str,
    calibration_dir: str = DEFAULT_APP_CALIBRATION_DIR,
    arena_dir: str = None,
    cam_name: str = None,
    image_date: str = None,
    keep_backup: bool = False,
    logger=None,
):
    """Reapply calibration to a single edited DLC pose file."""
    return run_reapply_calibration(
        pose_file=pose_file,
        video_path=video_path,
        calibration_dir=calibration_dir,
        arena_dir=arena_dir,
        cam_name=cam_name,
        image_date=image_date,
        keep_backup=keep_backup,
        logger=logger,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reapply calibration to the current edited pose file using app defaults."
    )
    parser.add_argument("--pose-file", required=True, help="Edited DLC parquet/csv path")
    parser.add_argument("--video-path", required=True, help="Matching video path")
    parser.add_argument(
        "--calibration-dir",
        default=DEFAULT_APP_CALIBRATION_DIR,
        help=f"Calibration directory (default: {DEFAULT_APP_CALIBRATION_DIR})",
    )
    parser.add_argument("--arena-dir", default=None, help="Optional Arena directory override")
    parser.add_argument("--cam-name", default=None, help="Optional camera name override")
    parser.add_argument("--image-date", default=None, help="Optional calibration date override")
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep/create a .pre_recalibration.bak file before overwriting",
    )
    args = parser.parse_args()

    run_edited_pose_calibration_job(
        pose_file=args.pose_file,
        video_path=args.video_path,
        calibration_dir=args.calibration_dir,
        arena_dir=args.arena_dir,
        cam_name=args.cam_name,
        image_date=args.image_date,
        keep_backup=bool(args.keep_backup),
        logger=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        sys.exit(1)
