#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Retrain DeepLabCut from manual labels (CLI wrapper for retrain_dlc_from_manual_labels.py).

You can provide either:
  1) --dlc-config /path/to/DLC/project/config.yaml
or
  2) --predict-config /path/to/Arena/configurations/predict_config.json
     (then we auto-locate the DLC training config via the "deeplabcut" model_path)

Required:
  --labels-root PATH          Manual labels root (images/train + labels/train)

Common:
  --video-path PATH           Optional; only needed if you want to (a) add the video into DLC config video_sets, or (b) run the optional re-run step
  --iterations N              Default: 5000
  --shuffle N                 Default: 1
  --net-type NAME             Optional; passed to DLC create_training_dataset (e.g. resnet_152)
  --augmenter-type NAME       Optional; passed to DLC create_training_dataset (e.g. imgaug)
  --posecfg-template PATH     Optional; pose_cfg.yaml template for create_training_dataset
  --init-weights PATH         Optional; checkpoint base path to fine-tune from (e.g. /path/to/snapshot-500000)

PreyTouch rerun (optional):
  --run-model-script PATH     Path to Arena/run_model.py
  --model-key KEY             Predict-config key (default: deeplabcut)
  --cam-name NAME             Default: top

Examples:
  # Retrain only (no video needed; uses labeled frames under --labels-root):
  ./retrain.sh \
    --predict-config /a/home/cc/students/neurosci/bareketd1/data/rep4/Arena/configurations/predict_config.json \
    --labels-root /path/to/manual_labels \
    --iterations 5000

  # Retrain + (optional) rerun a specific video:
  ./retrain.sh \
    --predict-config /a/home/cc/students/neurosci/bareketd1/data/rep4/Arena/configurations/predict_config.json \
    --labels-root /path/to/manual_labels \
    --video-path /path/to/video.mp4 \
    --iterations 5000

  ./retrain.sh \
    --dlc-config /path/to/DLC/project/config.yaml \
    --labels-root /path/to/manual_labels \
    --iterations 5000
EOF
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# DLC training does not depend on TensorRT. Hide TensorFlow's missing-libnvinfer warnings
# by default unless the caller explicitly asks for more verbose TF logs.
: "${TF_CPP_MIN_LOG_LEVEL:=2}"
export TF_CPP_MIN_LOG_LEVEL

PREDICT_CONFIG=""
DLC_CONFIG=""
MODEL_KEY="deeplabcut"
LABELS_ROOT=""
VIDEO_PATH=""
ITERS="5000"
SHUFFLE="1"
DATASET_NAME=""
RUN_MODEL_SCRIPT=""
CAM_NAME="top"
MODEL_PATH_OVERRIDE=""
NET_TYPE=""
AUGMENTER_TYPE=""
POSECFG_TEMPLATE=""
INIT_WEIGHTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predict-config) PREDICT_CONFIG="${2:-}"; shift 2 ;;
    --dlc-config) DLC_CONFIG="${2:-}"; shift 2 ;;
    --model-key) MODEL_KEY="${2:-}"; shift 2 ;;
    --labels-root) LABELS_ROOT="${2:-}"; shift 2 ;;
    --video-path) VIDEO_PATH="${2:-}"; shift 2 ;;
    --iterations|--iters) ITERS="${2:-}"; shift 2 ;;
    --shuffle) SHUFFLE="${2:-}"; shift 2 ;;
    --dataset-name) DATASET_NAME="${2:-}"; shift 2 ;;
    --net-type) NET_TYPE="${2:-}"; shift 2 ;;
    --augmenter-type) AUGMENTER_TYPE="${2:-}"; shift 2 ;;
    --posecfg-template) POSECFG_TEMPLATE="${2:-}"; shift 2 ;;
    --init-weights) INIT_WEIGHTS="${2:-}"; shift 2 ;;
    --run-model-script) RUN_MODEL_SCRIPT="${2:-}"; shift 2 ;;
    --cam-name) CAM_NAME="${2:-}"; shift 2 ;;
    --model-path) MODEL_PATH_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -z "$LABELS_ROOT" ]]; then
  echo "[ERROR] Missing --labels-root" >&2
  usage
fi

if [[ -z "$DLC_CONFIG" ]]; then
  if [[ -z "$PREDICT_CONFIG" ]]; then
    echo "[ERROR] Provide either --dlc-config or --predict-config" >&2
    usage
  fi

  DLC_CONFIG="$(
    "$PYTHON_BIN" - "$PREDICT_CONFIG" "$MODEL_KEY" <<'PY'
import json
import sys
from pathlib import Path

predict_config = Path(sys.argv[1]).expanduser().resolve()
model_key = sys.argv[2]

cfg = json.loads(predict_config.read_text(encoding="utf-8"))
if model_key not in cfg:
    keys = ", ".join(sorted(map(str, cfg.keys())))
    raise SystemExit(
        f"Model key '{model_key}' not in {predict_config}. Available keys: {keys}"
    )

entry = cfg.get(model_key)
if not isinstance(entry, dict):
    raise SystemExit(f"predict_config['{model_key}'] is not an object")

model_path_text = entry.get("model_path")
if not model_path_text:
    raise SystemExit(f"No model_path for '{model_key}' in {predict_config}")

mp = Path(model_path_text)
# predict_config is typically: <preytouch_root>/Arena/configurations/predict_config.json
# Map remote absolute paths like /data/PreyTouch/... to the local mount root.
if model_path_text.startswith("/data/PreyTouch/"):
    preytouch_root = predict_config.parents[2]
    mp = preytouch_root / model_path_text[len("/data/PreyTouch/") :]

# We expect a training project at: <...>/output/models/deeplabcut/train/<task>/config.yaml
train_dir = mp.parent / "train"
candidates = sorted(train_dir.glob("*/config.yaml"))

if len(candidates) == 1:
    print(str(candidates[0]))
    raise SystemExit(0)

fallback = mp.parent / "config.yaml"
if fallback.exists():
    print(str(fallback))
    raise SystemExit(0)

if len(candidates) == 0:
    raise SystemExit(
        f"Could not find DLC training config under {train_dir} (and no {fallback}).\n"
        f"model_path resolved to: {mp}"
    )

raise SystemExit(
    "Multiple DLC config.yaml candidates found. Specify --dlc-config explicitly:\n"
    + "\n".join(str(p) for p in candidates)
)
PY
  )"
fi

RETRAIN_SCRIPT="$SCRIPT_DIR/retrain_dlc_from_manual_labels.py"
if [[ ! -f "$RETRAIN_SCRIPT" ]]; then
  echo "[FATAL] Missing retrain script: $RETRAIN_SCRIPT" >&2
  exit 3
fi

cmd=(
  "$PYTHON_BIN" "$RETRAIN_SCRIPT"
  --dlc-config "$DLC_CONFIG"
  --labels-root "$LABELS_ROOT"
  --shuffle "$SHUFFLE"
  --iterations "$ITERS"
  --cam-name "$CAM_NAME"
)

if [[ -n "$VIDEO_PATH" ]]; then
  cmd+=( --video-path "$VIDEO_PATH" )
fi
if [[ -n "$DATASET_NAME" ]]; then
  cmd+=( --dataset-name "$DATASET_NAME" )
fi
if [[ -n "$NET_TYPE" ]]; then
  cmd+=( --net-type "$NET_TYPE" )
fi
if [[ -n "$AUGMENTER_TYPE" ]]; then
  cmd+=( --augmenter-type "$AUGMENTER_TYPE" )
fi
if [[ -n "$POSECFG_TEMPLATE" ]]; then
  cmd+=( --posecfg-template "$POSECFG_TEMPLATE" )
fi
if [[ -n "$INIT_WEIGHTS" ]]; then
  cmd+=( --init-weights "$INIT_WEIGHTS" )
fi
if [[ -n "$RUN_MODEL_SCRIPT" ]]; then
  cmd+=( --run-model-script "$RUN_MODEL_SCRIPT" --model-name "$MODEL_KEY" )
fi
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
  cmd+=( --model-path "$MODEL_PATH_OVERRIDE" )
fi

echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] DLC config: $DLC_CONFIG"
echo "[INFO] Labels root: $LABELS_ROOT"
echo "[INFO] Video path: ${VIDEO_PATH:-<none>}"
echo "[INFO] Iterations: $ITERS (shuffle=$SHUFFLE)"
echo "[INFO] Cam name: $CAM_NAME"
echo "[INFO] TF log level: $TF_CPP_MIN_LOG_LEVEL"

exec "${cmd[@]}"
