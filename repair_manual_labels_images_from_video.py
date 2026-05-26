#!/usr/bin/env python3
"""
Repair (recreate) manual_labels images by re-extracting the labeled frames from a video.

This is useful when manual_labels/images/train contains corrupted/truncated PNGs
(often due to partial copies or flaky network mounts), which can cause DeepLabCut
training to hang.

Inputs:
  - labels_root/labels/train/*.txt  (frame numbers inferred from filenames: *_f0000123.txt)
  - a video file containing those frames

Output:
  - labels_root/images/train/<same_stem>.png (overwritten)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def die(message: str) -> "None":
    print(f"[FATAL] {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_frame_from_label_stem(stem: str) -> Tuple[str, int]:
    m = re.search(r"^(?P<prefix>.+)_f(?P<frame>\d+)$", stem)
    if not m:
        raise ValueError(f"Label filename does not match '*_f0000000.txt' pattern: {stem}")
    return m.group("prefix"), int(m.group("frame"))


def safe_write_png(image_bgr, dst_png: Path):
    import cv2  # local import: only needed when actually running extraction

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = dst_png.with_name(dst_png.stem + ".tmp.png")

    ok = cv2.imwrite(str(tmp_png), image_bgr)
    if not ok or not tmp_png.exists():
        try:
            tmp_png.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"cv2.imwrite failed: {tmp_png}")

    if cv2.imread(str(tmp_png), cv2.IMREAD_UNCHANGED) is None:
        try:
            tmp_png.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Wrote unreadable image (likely truncated): {tmp_png}")

    tmp_png.replace(dst_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="Path to the source video (mp4/avi/...)")
    ap.add_argument(
        "--labels-root",
        required=True,
        help="manual_labels root (contains labels/train and images/train)",
    )
    ap.add_argument(
        "--video-stem",
        default=None,
        help="Optional filter prefix (e.g. 'top_20260226T143510'); defaults to stem of --video",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be regenerated; do not write images",
    )
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    labels_root = Path(args.labels_root).expanduser().resolve()
    if not video.exists():
        die(f"Video not found: {video}")
    if not labels_root.exists():
        die(f"labels-root not found: {labels_root}")

    labels_dir = labels_root / "labels" / "train"
    images_dir = labels_root / "images" / "train"
    if not labels_dir.exists():
        die(f"labels dir not found: {labels_dir}")

    filter_prefix = str(args.video_stem or video.stem)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        die(f"No label .txt files found in: {labels_dir}")

    frame_to_out: Dict[int, Path] = {}
    skipped_other_prefix = 0
    bad_name = 0
    for lp in label_files:
        try:
            prefix, frame_num = parse_frame_from_label_stem(lp.stem)
        except ValueError:
            bad_name += 1
            continue
        if prefix != filter_prefix:
            skipped_other_prefix += 1
            continue
        frame_to_out[frame_num] = images_dir / f"{lp.stem}.png"

    if not frame_to_out:
        die(
            f"No labels matched prefix '{filter_prefix}'. "
            f"label_files={len(label_files)}, bad_name={bad_name}, skipped_other_prefix={skipped_other_prefix}"
        )

    frames = sorted(frame_to_out.keys())
    max_frame = max(frames)
    print(f"[INFO] video:       {video}")
    print(f"[INFO] labels_root: {labels_root}")
    print(f"[INFO] prefix:      {filter_prefix}")
    print(f"[INFO] labels:      {len(frames)} frames (max_frame={max_frame})")
    if args.dry_run:
        for f in frames[:20]:
            print(f"[DRY] frame {f} -> {frame_to_out[f]}")
        more = "" if len(frames) <= 20 else f" (and {len(frames) - 20} more)"
        print(f"[DRY] done{more}")
        return

    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        die(f"Failed opening video with cv2: {video}")

    written = 0
    failed: List[Tuple[int, str]] = []
    try:
        needed = set(frame_to_out.keys())
        frame_idx = 0
        while frame_idx <= max_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in needed:
                out_png = frame_to_out[frame_idx]
                try:
                    safe_write_png(frame_bgr, out_png)
                    written += 1
                except Exception as exc:
                    failed.append((frame_idx, str(exc)))
                needed.remove(frame_idx)
                if written % 50 == 0:
                    print(f"[INFO] wrote {written} / {len(frames)} ...")
                if not needed:
                    break
            frame_idx += 1
    finally:
        cap.release()

    print(f"[INFO] wrote:  {written} / {len(frames)}")
    if failed:
        print(f"[ERROR] failed: {len(failed)}")
        for frame_idx, msg in failed[:10]:
            print(f"  - frame {frame_idx}: {msg}")
        raise SystemExit(2)

    if written != len(frames):
        missing = len(frames) - written
        print(f"[WARN] missing {missing} frames (video may be too short or read failed)")
        raise SystemExit(3)

    print("[DONE] manual_labels images regenerated successfully.")


if __name__ == "__main__":
    main()
