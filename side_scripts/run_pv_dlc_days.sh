#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PreyTouch


calib="/media/sil3/Data/Bareket/arena_configs/reptilearn5/calibrations"
model="/media/sil4/Data1/Bareket/models/deeplabcut/front_top_head_resnet_152_retrained"
cam="top"

base_video_folder="/media/sil2/Data/Bareket/experiments/reptilearn5/PV263/"
video_dates=(
  # "20260222"
  # "20260224"
  # "20260226"
  # "20260303"
  # "20260305"
  # "20260309"
  # "20260313"
  "20260609"
  "20260611"
  "20260519"  
  "20260521"  
  "20260525"  
  "20260527"  
  "20260529"  
  "20260601"  
  "20260603"  
  "20260605"  
  "20260610"
)

for day in "${video_dates[@]}"; do
    videos="${base_video_folder}${day}"
    echo $videos

    python ../run_model.py \
        -p "$model" \
        -v "$videos" \
        -c "$cam" \
        --calib_dir "$calib" \
        --skip_existing \
        # --start_x 7.59 --pix_cm 0.027604 --screen_y -4.3 # rep4
        # --start_x 5.59 --pix_cm 0.027604 --screen_y 0.06 # rep5
        --start_x 15.75 --pix_cm 0.0264 --screen_y 10.0 # retro compatible



        # --start_x 7.59 \
        # --pix_cm 0.027604 \
        # --screen_y -4.3
done