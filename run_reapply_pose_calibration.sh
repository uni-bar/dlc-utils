#!/bin/bash
POSE_FILE="/Volumes/Data/Bareket/experiments/reptilearn4/PV82/20260225/block8/videos/predictions/front_head_only_resnet_152__top_20260225T150051.parquet"
VIDEO_PATH="/Volumes/Data/Bareket/experiments/reptilearn4/PV82/20260225/block8/videos/top_20260225T150051.mp4"
CALIBRATION_DIR="/Volumes/Data/Bareket/arenas_configs/zeology low/calibrations"

PYTHONNOUSERSITE=1 /Users/bareketdamari/anaconda3/envs/PreyTouch/bin/python reapply_pose_calibration.py \
  --pose-file "$POSE_FILE" \
  --video-path "$VIDEO_PATH" \
  --calibration-dir "$CALIBRATION_DIR"
