#!/bin/bash
POSE_FILE="/media/sil3/Data/Bareket/experiments/reptilearn4/PV82/20260225/block8/videos/predictions/front_head_only_resnet_152__top_20260225T150051.parquet"
VIDEO_PATH="/media/sil3/Data/Bareket/experiments/reptilearn4/PV82/20260225/block8/videos/top_20260225T150051.mp4"
CALIBRATION_DIR="/media/sil3/Data/Bareket/arenas_configs/zeology low/calibrations"

PYTHONNOUSERSITE=1 python reapply_pose_calibration.py \
  --pose-file "$POSE_FILE" \
  --video-path "$VIDEO_PATH" \
  --calibration-dir "$CALIBRATION_DIR"
