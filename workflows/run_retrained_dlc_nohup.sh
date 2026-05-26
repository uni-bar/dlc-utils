#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage: run_retrained_dlc_nohup.sh [--detach] [--log-file PATH] [--skip-existing]

Options:
  --detach           Launch the workflow under nohup and exit immediately.
  --log-file PATH    Log file path for detached mode.
  --skip-existing    Pass --skip_existing to run_model.py for each date.
  -h, --help         Show this help.
EOF
}

DETACH_MODE=0
DETACHED_CHILD=0
DETACH_LOG_FILE=""
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH_MODE=1
      shift
      ;;
    --detached-child)
      DETACHED_CHILD=1
      shift
      ;;
    --log-file)
      DETACH_LOG_FILE="${2:-}"
      [[ -n "$DETACH_LOG_FILE" ]] || { echo "[FATAL] --log-file requires a path" >&2; exit 2; }
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[FATAL] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DETACH_MODE" == "1" && "$DETACHED_CHILD" == "1" ]]; then
  echo "[FATAL] --detach and --detached-child cannot be used together." >&2
  exit 2
fi

if [[ "$DETACH_MODE" == "1" ]]; then
  LOG_DIR="$SCRIPT_DIR/logs"
  mkdir -p "$LOG_DIR"
  if [[ -z "$DETACH_LOG_FILE" ]]; then
    DETACH_LOG_FILE="$LOG_DIR/run_retrained_dlc_nohup_$(date +%Y%m%d_%H%M%S).log"
  elif [[ "$DETACH_LOG_FILE" != /* ]]; then
    DETACH_LOG_FILE="$PWD/$DETACH_LOG_FILE"
  fi
  mkdir -p "$(dirname "$DETACH_LOG_FILE")"

  launch_args=(--detached-child)
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    launch_args+=(--skip-existing)
  fi
  printf -v launch_cmd '%q ' "$SCRIPT_PATH" "${launch_args[@]}"

  nohup bash -lc "$launch_cmd" >"$DETACH_LOG_FILE" 2>&1 < /dev/null &
  pid=$!

  echo "[INFO] Detached run started."
  echo "[INFO] PID:      $pid"
  echo "[INFO] Log file: $DETACH_LOG_FILE"
  echo "[INFO] Follow:   tail -f \"$DETACH_LOG_FILE\""
  exit 0
fi

if [[ "$DETACHED_CHILD" == "1" ]]; then
  echo "[INFO] Detached session started at $(date '+%F %T %Z')"
fi

# ARENA_DIR="$HOME/data/rep4/Arena"
DATA_ROOT_DIR="$HOME/media/Bareket/experiments/reptilearn4/PV82"
ALT_DATA_ROOT_DIR="$HOME/media/sil3/Bareket/experiments/reptilearn4/PV82"
# CAM_NAME="top"
# DATE_FOLDERS=({20260217..20260227})
ARENA_DIR="$HOME/data/rep4/Arena"
# ALT_DATA_ROOT_DIR="$HOME/media/sil3/Bareket/experiments/reptilearn5/PV252"

# DATA_ROOT_DIR="$HOME/media/sil3/Bareket/experiments/reptilearn5/PV252"
# DATA_ROOT_DIR="$HOME/data/rep4/output/experiments/PV82"
# ALT_DATA_ROOT_DIR="$HOME/media/sil3/Bareket/experiments/reptilearn5/PV252"
CAM_NAME="top"
# DATE_FOLDERS=({20260223..20260313})

DATE_FOLDERS=({20260218..20260227})

CONDA_MODULE="miniconda"
CONDA_ENV_DIR="/scratch200/bareketd1/PreyTouch"
MODEL_NAME="deeplabcut"
MODEL_OVERRIDE_PATH="$HOME/data/rep4/output/models/deeplabcut/front_top_head_resnet_152_retrained"

# MODEL_OVERRIDE_PATH="$HOME/data/rep4/output/models/deeplabcut/front_top_head_resnet_152_retrained"

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

[[ -d "$ARENA_DIR" ]] || { echo "[FATAL] ARENA_DIR not found: $ARENA_DIR" >&2; exit 1; }
[[ -f "$ARENA_DIR/run_model.py" ]] || { echo "[FATAL] run_model.py not found under ARENA_DIR: $ARENA_DIR" >&2; exit 1; }
[[ -d "$MODEL_OVERRIDE_PATH" ]] || { echo "[FATAL] MODEL_OVERRIDE_PATH not found: $MODEL_OVERRIDE_PATH" >&2; exit 1; }
[[ -f "$MODEL_OVERRIDE_PATH/pose_cfg.yaml" ]] || { echo "[FATAL] pose_cfg.yaml not found under MODEL_OVERRIDE_PATH: $MODEL_OVERRIDE_PATH" >&2; exit 1; }

if [[ -d "$DATA_ROOT_DIR" ]]; then
  :
elif [[ -d "$ALT_DATA_ROOT_DIR" ]]; then
  DATA_ROOT_DIR="$ALT_DATA_ROOT_DIR"
else
  echo "[FATAL] Neither DATA_ROOT_DIR nor ALT_DATA_ROOT_DIR exists:" >&2
  echo "  - $DATA_ROOT_DIR" >&2
  echo "  - $ALT_DATA_ROOT_DIR" >&2
  echo "[HINT] If this is an sshfs mount, try: ~/scripts/utils/mount_sils.sh" >&2
  exit 1
fi

cd "$ARENA_DIR"

echo "[INFO] Loading module: $CONDA_MODULE"
module load "$CONDA_MODULE" >/dev/null
echo "[INFO] Activating conda env: $CONDA_ENV_DIR"
conda activate "$CONDA_ENV_DIR"

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

export IS_USE_REDIS=0

echo "[INFO] arena_dir:      $ARENA_DIR"
echo "[INFO] model_override: $MODEL_OVERRIDE_PATH"
echo "[INFO] data_root:      $DATA_ROOT_DIR"
echo "[INFO] cam_name:       $CAM_NAME"
echo "[INFO] dates:          ${DATE_FOLDERS[*]}"

ran_any=0
for date_folder in "${DATE_FOLDERS[@]}"; do
  data_dir="$DATA_ROOT_DIR/$date_folder"
  if [[ ! -d "$data_dir" ]]; then
    echo "[WARN] Date folder not found (skipping): $data_dir" >&2
    continue
  fi

  video_match="$(find "$data_dir" -type f \( -name "${CAM_NAME}*.mp4" -o -name "${CAM_NAME}*.avi" \) -print -quit 2>/dev/null || true)"
  if [[ -z "$video_match" ]]; then
    echo "[WARN] No ${CAM_NAME} videos found under (skipping): $data_dir" >&2
    continue
  fi

  ran_any=1
  args=(
    -y -m "$MODEL_NAME" -c "$CAM_NAME" -p "$data_dir"
    --video_suffixes .mp4
    --model_override "$MODEL_OVERRIDE_PATH"
    --start_x "$START_X" --pix_cm "$PIX_CM" --screen_y "$SCREEN_Y"
  )
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    args+=(--skip_existing)
  fi

  python run_model.py "${args[@]}"
done

if [[ "$ran_any" != "1" ]]; then
  echo "[WARN] No matching date folders/videos found; nothing to do." >&2
  exit 0
fi
