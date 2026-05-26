#!/usr/bin/env bash
set -euo pipefail

# Retrain preset for:
#  - labels:    ~/sandbox/dlc-utils/manual_labels
#  - model:     front_head_only_resnet_152  ->  front_head_only_resnet_152_retrained
#
# IMPORTANT: this script needs a Python that can import DeepLabCut.
# You asked to use the PreyTouch env at /scratch200/bareketd1/PreyTouch, so that is the default.

# =========================
# User Params (edit here)
# =========================
PREYTOUCH_ENV="/scratch200/bareketd1/PreyTouch"
PYTHON_BIN="$PREYTOUCH_ENV/bin/python"

# TensorFlow 2.10 in this env is a CUDA build (expects CUDA 11.x + cuDNN 8).
# On the HPC, load matching modules so GPU training doesn't crash.
CUDA_MODULE="CUDA/CUDA-11.8"
CUDNN_MODULE="cudnn/cudnn-8.3.2.4"
TENSORRT_MODULE="${TENSORRT_MODULE:-}"   # optional; set if your HPC exposes a matching TensorRT module
TENSORRT_LIB_DIR="${TENSORRT_LIB_DIR:-}" # optional; prepend a manual TensorRT lib dir to LD_LIBRARY_PATH
REQUIRE_GPU="1"       # set to 0 to allow CPU training
RUN_TF_SMOKE_TEST="1" # set to 0 to skip quick GPU sanity check
PY_IMPORT_TIMEOUT_SECS="${PY_IMPORT_TIMEOUT_SECS:-45}"
TF_SMOKE_TIMEOUT_SECS="${TF_SMOKE_TIMEOUT_SECS:-90}"

# DLC training does not require TF-TRT. Hide the noisy "missing libnvinfer" warnings
# unless the user explicitly wants full TensorFlow warning output.
: "${TF_CPP_MIN_LOG_LEVEL:=2}"
export TF_CPP_MIN_LOG_LEVEL

LABELS_ROOT="${LABELS_ROOT:-$HOME/sandbox/dlc-utils/manual_labels}"

# Source DLC training project (on sshfs mount; contains labeled-data/, training-datasets/, dlc-models/, exported-models/)
SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR:-/a/home/cc/students/neurosci/bareketd1/data/rep4/output/models/deeplabcut/train/front_head_only}"
SOURCE_CONFIG_YAML="${SOURCE_CONFIG_YAML:-$SOURCE_PROJECT_DIR/config.yaml}"
# Used to keep create_training_dataset settings consistent with the existing project.
# If this exact path doesn't exist (project changed), we'll auto-detect a suitable pose_cfg.yaml below.
SOURCE_POSECFG_TEMPLATE="${SOURCE_POSECFG_TEMPLATE:-$SOURCE_PROJECT_DIR/dlc-models/iteration-2/front_head_onlyJan23-trainset95shuffle2/train/pose_cfg.yaml}"

# Work copy (writable scratch; retraining happens here)
WORK_PROJECT_DIR="${WORK_PROJECT_DIR:-/scratch200/bareketd1/tmp/dlc_retrain/front_head_only}"
WORK_CONFIG_YAML="${WORK_CONFIG_YAML:-$WORK_PROJECT_DIR/config_hpc_retrain.yaml}"

# Output model folder (copied from DLC exported-models)
OUT_MODEL_DIR="${OUT_MODEL_DIR:-/a/home/cc/students/neurosci/bareketd1/data/rep4/output/models/deeplabcut/front_head_only_resnet_152_retrained}"

# Base model to fine-tune from (exported model folder containing snapshot-*.index)
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/a/home/cc/students/neurosci/bareketd1/data/rep4/output/models/deeplabcut/front_head_only_resnet_152}"

# Training knobs
SHUFFLE="2"   # shuffle=2 is the resnet_152 model for this project
ITERS="5000"  # DLC train_network maxiters
NET_TYPE="resnet_152"
AUGMENTER_TYPE="imgaug"
PRUNE_MISSING_IMAGE_LABELS="${PRUNE_MISSING_IMAGE_LABELS:-1}"
PRUNE_BAD_IMAGES="${PRUNE_BAD_IMAGES:-1}"
SKIP_BAD_IMAGES="${SKIP_BAD_IMAGES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$SCRIPT_DIR/retrain_dlc_from_manual_labels.py" ]]; then
  RETRAIN_HELPER="$SCRIPT_DIR/retrain_dlc_from_manual_labels.py"
else
  RETRAIN_HELPER="$UTILS_DIR/retrain_dlc_from_manual_labels.py"
fi

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

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

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module_args=("$CUDA_MODULE" "$CUDNN_MODULE")
    if [[ -n "$TENSORRT_MODULE" ]]; then
      module_args+=("$TENSORRT_MODULE")
    fi
    echo "[INFO] Loading HPC modules: ${module_args[*]}"
    module purge >/dev/null 2>&1 || true
    module load "${module_args[@]}"
  else
    echo "[WARN] 'module' command not found; relying on existing CUDA libs in LD_LIBRARY_PATH."
  fi

  if [[ -n "$TENSORRT_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$TENSORRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[INFO] Added TensorRT libs to LD_LIBRARY_PATH: $TENSORRT_LIB_DIR"
  fi

  # Respect scheduler/user GPU selection. If none is set, pick the GPU with the most free memory.
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(pick_gpu_with_most_free_memory)" \
      || die "Unable to auto-select a GPU. Set CUDA_VISIBLE_DEVICES explicitly."
    [[ -n "$CUDA_VISIBLE_DEVICES" ]] \
      || die "Unable to auto-select a GPU. Set CUDA_VISIBLE_DEVICES explicitly."
    echo "[INFO] Auto-selected GPU: $CUDA_VISIBLE_DEVICES"
  else
    echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
  export CUDA_VISIBLE_DEVICES

  : "${TF_FORCE_GPU_ALLOW_GROWTH:=true}"
  export TF_FORCE_GPU_ALLOW_GROWTH
fi

echo "[INFO] labels-root:      $LABELS_ROOT"
echo "[INFO] source project:   $SOURCE_PROJECT_DIR"
echo "[INFO] work project:     $WORK_PROJECT_DIR"
echo "[INFO] source config:    $SOURCE_CONFIG_YAML"
echo "[INFO] work config:      $WORK_CONFIG_YAML"
echo "[INFO] output model dir: $OUT_MODEL_DIR"
echo "[INFO] iters/shuffle:    $ITERS / $SHUFFLE"
echo "[INFO] net/augmenter:    $NET_TYPE / $AUGMENTER_TYPE"
echo "[INFO] prune/skip-bad:   $PRUNE_MISSING_IMAGE_LABELS / $PRUNE_BAD_IMAGES / $SKIP_BAD_IMAGES"
echo "[INFO] base model dir:   $BASE_MODEL_DIR"
echo "[INFO] python:           $PYTHON_BIN"
echo "[INFO] TF log level:     $TF_CPP_MIN_LOG_LEVEL"
echo "[INFO] import timeout:   ${PY_IMPORT_TIMEOUT_SECS}s"
echo "[INFO] smoke timeout:    ${TF_SMOKE_TIMEOUT_SECS}s"

[[ -x "$PYTHON_BIN" ]] || die "Python not found/executable: $PYTHON_BIN"
[[ -d "$LABELS_ROOT" ]] || die "labels-root not found: $LABELS_ROOT"
[[ -f "$SOURCE_CONFIG_YAML" ]] || die "training config.yaml not found: $SOURCE_CONFIG_YAML"
[[ -d "$BASE_MODEL_DIR" ]] || die "base model dir not found: $BASE_MODEL_DIR"
[[ -f "$RETRAIN_HELPER" ]] || die "missing retrain helper: $RETRAIN_HELPER"

if [[ ! -f "$SOURCE_POSECFG_TEMPLATE" ]]; then
  echo "[WARN] pose_cfg template not found at configured path; auto-detecting latest shuffle${SHUFFLE} pose_cfg.yaml under dlc-models/..."
  SOURCE_POSECFG_TEMPLATE="$(
    "$PYTHON_BIN" - "$SOURCE_PROJECT_DIR" "$SHUFFLE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
shuffle = str(sys.argv[2])

candidates = list(root.glob(f"dlc-models/iteration-*/*shuffle{shuffle}/train/pose_cfg.yaml"))
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise SystemExit(f"No pose_cfg.yaml found under {root}/dlc-models for shuffle{shuffle}")
print(str(candidates[0]))
PY
  )" || die "Failed auto-detecting pose_cfg.yaml under: $SOURCE_PROJECT_DIR/dlc-models"
fi
[[ -f "$SOURCE_POSECFG_TEMPLATE" ]] || die "pose_cfg template not found: $SOURCE_POSECFG_TEMPLATE"
echo "[INFO] posecfg template: $SOURCE_POSECFG_TEMPLATE"

# manual_labels can be either:
#  - pooled:    $LABELS_ROOT/images/train + $LABELS_ROOT/labels/train
#  - per-video: $LABELS_ROOT/<video>/images/train + $LABELS_ROOT/<video>/labels/train
# The retrain helper will merge legacy per-video exports into the pooled layout at runtime.
label_count="$(
  (
    find "$LABELS_ROOT" \
      \( -path '*/bckup' -o -path '*/bckup/*' \) -prune -o \
      -type f -path '*/labels/train/*.txt' -print 2>/dev/null || true
  ) | wc -l | tr -d ' '
)"
img_count="$(
  (
    find "$LABELS_ROOT" \
      \( -path '*/bckup' -o -path '*/bckup/*' \) -prune -o \
      -type f -path '*/images/train/*' \
      \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.tif' -o -name '*.tiff' -o -name '*.bmp' \) \
      -print 2>/dev/null || true
  ) | wc -l | tr -d ' '
)"
echo "[INFO] manual labels (any source): ${label_count} txt, ${img_count} images"
(( label_count > 0 )) || die "no label txt files found under $LABELS_ROOT/**/labels/train"

echo "[INFO] Checking Python/DLC availability..."
set +e
timeout "$PY_IMPORT_TIMEOUT_SECS" "$PYTHON_BIN" - <<'PY'
import sys
print("python:", sys.version.split()[0])
import deeplabcut  # noqa: F401
print("deeplabcut: import OK")
PY
py_rc="$?"
set -e
if [[ "$py_rc" -ne 0 ]]; then
  if [[ "$py_rc" -eq 124 ]]; then
    die "DeepLabCut/TensorFlow import timed out after ${PY_IMPORT_TIMEOUT_SECS}s in $PYTHON_BIN. This node/env is not healthy right now; rerun on a different node or retry later."
  fi
  die "DeepLabCut import failed in $PYTHON_BIN. For Python 3.8, install a DLC 2.x build (e.g. deeplabcut==2.3.11) into this env."
fi

if [[ "$REQUIRE_GPU" == "1" && "$RUN_TF_SMOKE_TEST" == "1" ]]; then
  echo "[INFO] TensorFlow GPU smoke test (detect + tiny op)..."
  set +e
  timeout "$TF_SMOKE_TIMEOUT_SECS" "$PYTHON_BIN" - <<'PY'
import ctypes
import sys

missing = []
for lib in ["libcudart.so.11.0", "libcublas.so.11", "libcudnn.so.8"]:
    try:
        ctypes.CDLL(lib)
    except OSError as e:
        missing.append(f"{lib} ({e})")
if missing:
    print("[FATAL] Missing CUDA libraries for TensorFlow:", file=sys.stderr)
    for m in missing:
        print("  -", m, file=sys.stderr)
    raise SystemExit(2)

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("tf:", tf.__version__)
print("gpus:", gpus)
if not gpus:
    print("[FATAL] No GPUs detected. Are you on a GPU node/allocation?", file=sys.stderr)
    raise SystemExit(3)

with tf.device("/GPU:0"):
    x = tf.random.normal([64, 64])
    y = tf.reduce_sum(x)
print("gpu_smoke_sum:", float(y.numpy()))
print("gpu_smoke_test: OK")
PY
  smoke_rc="$?"
  set -e
  if [[ "$smoke_rc" -ne 0 ]]; then
    if [[ "$smoke_rc" -eq 124 ]]; then
      die "TensorFlow GPU smoke test timed out after ${TF_SMOKE_TIMEOUT_SECS}s. This node/env is hanging during TF startup; rerun on a different node or retry later."
    fi
    die "TensorFlow GPU smoke test failed (rc=$smoke_rc). Most common fix: ensure CUDA 11.x + cuDNN 8 modules are loaded (e.g. $CUDA_MODULE + $CUDNN_MODULE)."
  fi
fi

[[ "$WORK_PROJECT_DIR" == /scratch200/* ]] || die "WORK_PROJECT_DIR must be under /scratch200 (got: $WORK_PROJECT_DIR)"
[[ "$WORK_PROJECT_DIR" != "$SOURCE_PROJECT_DIR" ]] || die "WORK_PROJECT_DIR must differ from SOURCE_PROJECT_DIR"

echo "[INFO] Syncing source project -> scratch (rsync)..."
mkdir -p "$WORK_PROJECT_DIR"
rsync -a --no-owner --no-group --delete \
  --exclude '/exported-models/' \
  --exclude '/evaluation-results/' \
  --exclude '/dlc-models/iteration-0/' \
  --exclude '/dlc-models/iteration-1/' \
  --exclude '/dlc-models/iteration-2/*shuffle1*/' \
  --exclude '*/train/log.txt' \
  --exclude '*/train/log/' \
  --exclude '*/test/log.txt' \
  --exclude '*/test/log/' \
  "$SOURCE_PROJECT_DIR"/ "$WORK_PROJECT_DIR"/

# Patch project_path into a config that lives INSIDE the *work* project folder.
echo "[INFO] Writing work config with patched project_path: $WORK_CONFIG_YAML"
"$PYTHON_BIN" - "$WORK_PROJECT_DIR/config.yaml" "$WORK_CONFIG_YAML" "$WORK_PROJECT_DIR" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
project_path = Path(sys.argv[3]).expanduser().resolve()

lines = src.read_text(encoding="utf-8").splitlines(True)
out = []
replaced = False
for ln in lines:
    if ln.lstrip().startswith("project_path:"):
        indent = ln[: len(ln) - len(ln.lstrip())]
        out.append(f"{indent}project_path: {project_path.as_posix()}\n")
        replaced = True
    else:
        out.append(ln)

if not replaced:
    out.append(f"\nproject_path: {project_path.as_posix()}\n")

dst.write_text("".join(out), encoding="utf-8")
print(str(dst))
PY

echo "[INFO] Detecting latest snapshot under base model dir..."
init_weights="$("$PYTHON_BIN" - "$BASE_MODEL_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
paths = list(root.glob("snapshot-*.index"))
best = None
best_n = -1
for p in paths:
    m = re.search(r"snapshot-(\d+)\.index$", p.name)
    if not m:
        continue
    n = int(m.group(1))
    if n > best_n:
        best_n = n
        best = p
if best is None:
    raise SystemExit(f"No snapshot-*.index found under {root}")
print(str(best).replace(".index", ""))
PY
)"
echo "[INFO] init_weights:     $init_weights"

echo "[INFO] Starting retrain (this can take a while)..."
helper_args=(
  --dlc-config "$WORK_CONFIG_YAML"
  --labels-root "$LABELS_ROOT"
  --shuffle "$SHUFFLE"
  --iterations "$ITERS"
  --net-type "$NET_TYPE"
  --augmenter-type "$AUGMENTER_TYPE"
  --posecfg-template "$SOURCE_POSECFG_TEMPLATE"
  --init-weights "$init_weights"
)

if [[ "$PRUNE_MISSING_IMAGE_LABELS" == "1" ]]; then
  helper_args+=(--prune-missing-image-labels)
fi
if [[ "$PRUNE_BAD_IMAGES" == "1" ]]; then
  helper_args+=(--prune-bad-images)
fi
if [[ "$SKIP_BAD_IMAGES" == "1" ]]; then
  helper_args+=(--skip-bad-images)
fi

"$PYTHON_BIN" "$RETRAIN_HELPER" "${helper_args[@]}"

exported_models_dir="$WORK_PROJECT_DIR/exported-models"
[[ -d "$exported_models_dir" ]] || die "No exported-models dir found after training: $exported_models_dir"

echo "[INFO] Finding latest exported resnet_152 model under: $exported_models_dir"
exported_model_dir="$("$PYTHON_BIN" - "$exported_models_dir" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
pose_cfgs = list(root.rglob("pose_cfg.yaml"))
pose_cfgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

for pose_cfg in pose_cfgs:
    parent = pose_cfg.parent
    name = parent.name.lower()
    # Prefer resnet_152 exports when present.
    if "resnet_152" not in name:
        continue
    if any(parent.glob("snapshot*.index")):
        print(str(parent))
        raise SystemExit(0)

# Fallback: any exported model with snapshots.
for pose_cfg in pose_cfgs:
    parent = pose_cfg.parent
    if any(parent.glob("snapshot*.index")):
        print(str(parent))
        raise SystemExit(0)

raise SystemExit(f"No exported model folder with pose_cfg.yaml + snapshot*.index under {root}")
PY
)"

echo "[INFO] Exported model folder: $exported_model_dir"
echo "[INFO] Syncing exported model -> $OUT_MODEL_DIR"
mkdir -p "$OUT_MODEL_DIR"
rsync -a --delete "$exported_model_dir"/ "$OUT_MODEL_DIR"/

echo "[DONE] Retrained model copied to:"
echo "  $OUT_MODEL_DIR"
echo
echo "[NEXT] To use it in PreyTouch, change:"
echo "  /data/PreyTouch/output/models/deeplabcut/$(basename "$BASE_MODEL_DIR")"
echo "to:"
echo "  /data/PreyTouch/output/models/deeplabcut/$(basename "$OUT_MODEL_DIR")"
