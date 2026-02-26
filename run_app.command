#!/bin/bash
# Simple launcher for DLC Video Overlay Tool
# Double-click this file in Finder to run the app!

cd "$(dirname "$0")"

# Activate conda environment if available
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate PTL 2>/dev/null
fi

# Try to find Python
if [ -f "$HOME/anaconda3/envs/PTL/bin/python" ]; then
    "$HOME/anaconda3/envs/PTL/bin/python" dlc_video_overlay.py
elif command -v python3 &> /dev/null; then
    python3 dlc_video_overlay.py
elif command -v python &> /dev/null; then
    python dlc_video_overlay.py
else
    osascript -e 'tell app "System Events" to display dialog "Python not found! Please install Python 3 or Anaconda." buttons {"OK"} default button 1'
fi
