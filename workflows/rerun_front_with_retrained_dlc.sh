#!/usr/bin/env bash
set -euo pipefail

# =========================
# User Params (edit here)
# =========================
PREYTOUCH_ENV="/scratch200/bareketd1/PreyTouch"
PYTHON_BIN="$PREYTOUCH_ENV/bin/python"

# TensorFlow in this env expects CUDA 11.x + cuDNN 8
CUDA_MODULE="CUDA/CUDA-11.8"
CUDNN_MODULE="cudnn/cudnn-8.3.2.4"

# Rep4 root (use $HOME so it works on both HPC + remote mounts)
REP4_ROOT="$HOME/data/rep4"
# If you run directly on the remote machine (no sshfs mount), rep4 is usually here:
ALT_REP4_ROOT="/data/PreyTouch"

# PreyTouch Arena (run_model.py lives here)
ARENA_DIR="$REP4_ROOT/Arena"

# Your new retrained model folder (exported DLC model: pose_cfg.yaml + snapshots)
MODEL_OVERRIDE_PATH="$REP4_ROOT/output/models/deeplabcut/front_head_only_resnet_152_retrained"

# Which configured model key to use from Arena/configurations/predict_config.json.
# With --model_override below, this only selects the predictor type; the actual weights path comes from MODEL_OVERRIDE_PATH.
MODEL_NAME="deeplabcut"

# Where your videos are (must exist on the machine you run this on)
DATA_ROOT_DIR="$HOME/media/Bareket/experiments/reptilearn4/PV82"
ALT_DATA_ROOT_DIR="$HOME/media/sil3/Bareket/experiments/reptilearn4/PV82"

# Date folders to run (under DATA_ROOT_DIR / ALT_DATA_ROOT_DIR)
DATE_FOLDERS=("20260122" "20260124" "20260125" "20260126")
CAM_NAME="front"

# These are REQUIRED by data/rep4/Arena/run_model.py (it calls float(...) on them)
START_X="15.75"
PIX_CM="0.0264"
SCREEN_Y="10.0"

# If set to 1, skips videos that already have a predictions parquet for this model.
SKIP_EXISTING="0"

# =========================
# Run
# =========================
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] Python not found: $PYTHON_BIN" >&2; exit 1; }

# If we're on a machine where $HOME/data/rep4 isn't mounted, fall back to /data/PreyTouch
if [[ ! -d "$ARENA_DIR" && -d "$ALT_REP4_ROOT/Arena" ]]; then
  REP4_ROOT="$ALT_REP4_ROOT"
  ARENA_DIR="$REP4_ROOT/Arena"
  MODEL_OVERRIDE_PATH="$REP4_ROOT/output/models/deeplabcut/front_head_only_resnet_152_retrained"
fi

[[ -d "$ARENA_DIR" ]] || { echo "[FATAL] ARENA_DIR not found: $ARENA_DIR" >&2; exit 1; }
[[ -d "$MODEL_OVERRIDE_PATH" ]] || { echo "[FATAL] MODEL_OVERRIDE_PATH not found: $MODEL_OVERRIDE_PATH" >&2; exit 1; }
[[ -f "$MODEL_OVERRIDE_PATH/pose_cfg.yaml" ]] || { echo "[FATAL] pose_cfg.yaml not found under MODEL_OVERRIDE_PATH: $MODEL_OVERRIDE_PATH" >&2; exit 1; }
[[ -d "$DATA_ROOT_DIR" || -d "$ALT_DATA_ROOT_DIR" ]] || { echo "[FATAL] Neither DATA_ROOT_DIR nor ALT_DATA_ROOT_DIR exists" >&2; exit 1; }

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load "$CUDA_MODULE" "$CUDNN_MODULE"
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
: "${TF_FORCE_GPU_ALLOW_GROWTH:=true}"
export TF_FORCE_GPU_ALLOW_GROWTH

# On compute nodes there's typically no local Redis server. Disable redis publishing to avoid noisy stack traces.
: "${IS_USE_REDIS:=0}"
export IS_USE_REDIS

cd "$ARENA_DIR"

echo "[INFO] rep4_root:      $REP4_ROOT"
echo "[INFO] model_override: $MODEL_OVERRIDE_PATH"
echo "[INFO] cam_name:       $CAM_NAME"
echo "[INFO] dates:          ${DATE_FOLDERS[*]}"
echo "[INFO] Running predictions (this writes *.parquet under each video's ./predictions/ folder)"

ran_any=0
for date_folder in "${DATE_FOLDERS[@]}"; do
  data_dir="$DATA_ROOT_DIR/$date_folder"
  alt_data_dir="$ALT_DATA_ROOT_DIR/$date_folder"
  if [[ ! -d "$data_dir" && -d "$alt_data_dir" ]]; then
    data_dir="$alt_data_dir"
  fi
  if [[ ! -d "$data_dir" ]]; then
    echo "[WARN] Date folder not found (skipping): $date_folder" >&2
    continue
  fi

  # Skip if there are no matching videos under this date folder (avoid loading the model for nothing).
  if ! find "$data_dir" -type f \( -name "${CAM_NAME}*.mp4" -o -name "${CAM_NAME}*.avi" \) -print -quit >/dev/null 2>&1; then
    echo "[WARN] No ${CAM_NAME} videos found under (skipping): $data_dir" >&2
    continue
  fi

  echo
  echo "[INFO] data_dir:       $data_dir"
  args=(
    -y -m "$MODEL_NAME"
    -c "$CAM_NAME"
    -p "$data_dir"
    --model_override "$MODEL_OVERRIDE_PATH"
    --start_x "$START_X" --pix_cm "$PIX_CM" --screen_y "$SCREEN_Y"
  )
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    args+=(--skip_existing)
  fi

  ran_any=1
  "$PYTHON_BIN" run_model.py "${args[@]}"
done

if [[ "$ran_any" != "1" ]]; then
  echo "[WARN] No matching date folders/videos found; nothing to do." >&2
  exit 0
fi
