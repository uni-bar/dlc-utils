#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID=$(date +%Y%m%d_%H%M%S)

LABELS_ROOT=$HOME/sandbox/dlc-utils/manual_labels \
SOURCE_PROJECT_DIR=/scratch200/bareketd1/tmp/dlc_retrain/front_head_only \
WORK_PROJECT_DIR=/scratch200/bareketd1/tmp/dlc_retrain/front_head_only_$RUN_ID \
BASE_MODEL_DIR=/a/home/cc/students/neurosci/bareketd1/data/rep4/output/models/deeplabcut/front_top_head_resnet_152_retrained \
OUT_MODEL_DIR=/a/home/cc/students/neurosci/bareketd1/data/rep4/output/models/deeplabcut/front_top_head_resnet_152_retrained \
bash "$SCRIPT_DIR/retrain_front_head_only_resnet_152_retrained.sh" |& tee retrain_${RUN_ID}.log


# OUT_MODEL_DIR=/a/home/cc/students/neurosci/bareketd1/data/rep5/output/models/deeplabcut/top_head_resnet_152_retrained_5 \
