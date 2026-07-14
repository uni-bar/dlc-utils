#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PreyTouch


calib="/media/sil3/Data/Bareket/arena_configs/reptilearn5/calibrations"
# model="/media/sil4/Data1/Bareket/models/deeplabcut/front_top_head_resnet_152_retrained"
cam="top"

# CSV_FILE="/home/bareket/sandbox/.cache/PV163/pv163_missing_blocks.csv"    #"test.csv"
CSV_FILE="./test.csv"
model="new_models"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r lab hpc; do
    echo "Running on: $lab"
    python ../run_model.py \
    -p "$model" \
    -v "$lab" \
    -c "$cam" \
    --calib_dir "$calib" \
    --skip_existing \
    #--start_x 7.59 --pix_cm 0.027604 --screen_y -4.3 # rep4
    --start_x 5.59 --pix_cm 0.027604 --screen_y 0.06 # rep5

done
