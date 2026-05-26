#!/usr/bin/env bash
set -euo pipefail

# data_dir="$HOME/media/sil3/Bareket/experiments/reptilearn4/PV82/20260225/"
data_dir="$HOME/media/sil3/Bareket/experiments/reptilearn4/PV82/20260308/" #block3/videos/
#  /Volumes/Data/Bareket/experiments/reptilearn4/PV82/20260225/block3/videos/top_20260225T123542.mp4

ARENA_DIR="$HOME/data/rep4/Arena"
MODEL_NAME="deeplabcut-top"
MODEL_OVERRIDE_PATH="$HOME/data/rep4/output/models/deeplabcut/front_top_head_resnet_152_retrained"
CAM_NAME="top"
CALIBRATION_DIR="$HOME/data/rep4/output/calibrations"

START_X="15.75"
PIX_CM="0.0264"
SCREEN_Y="10.0"

pick_gpu_with_most_free_memory() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '
        BEGIN {best_idx=""; best_free=-1}
        {
          idx=$1
          free=$2
          gsub(/^[ \t]+|[ \t]+$/, "", idx)
          gsub(/^[ \t]+|[ \t]+$/, "", free)
          if (free+0 > best_free) {
            best_free = free+0
            best_idx = idx
          }
        }
        END {
          if (best_idx != "") {
            print best_idx
          }
        }'
}

cd "$ARENA_DIR"
module load miniconda >/dev/null
conda activate /scratch200/sheinmark/envs/PreyTouch

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(pick_gpu_with_most_free_memory)" || {
    echo "[WARN] Unable to auto-select a GPU; leaving CUDA_VISIBLE_DEVICES unset." >&2
    CUDA_VISIBLE_DEVICES=""
  }
  if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "[INFO] Auto-selected GPU: $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES
  fi
else
  echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  export CUDA_VISIBLE_DEVICES
fi

python run_model.py -y -m "$MODEL_NAME" -c "$CAM_NAME" -p "$data_dir" \
    --video_suffixes .mp4 \
    --model_override "$MODEL_OVERRIDE_PATH" \
    --start_x "$START_X" --pix_cm "$PIX_CM" --screen_y "$SCREEN_Y" \
    --calib_dir "$CALIBRATION_DIR"
