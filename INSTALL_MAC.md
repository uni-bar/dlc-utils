# Installation Guide for Mac App

## Changes Made

### ✅ Removed:
- "Flip Y-axis" checkbox - No longer needed, coordinates are used as-is

### ✅ Added:
- **Playback Speed Control** - Slider to adjust video speed from 0.1x to 2.0x
  - Located in left panel under "Playback Speed"
  - Default is 1.0x (normal speed)
  - Adjust while playing or paused to see video in slow-motion or fast-forward

## Running the Tool

### Option 1: Run with Python (Easiest)
```bash
cd /Users/bareketdamari/Dev/PreyTouch/sandbox/videos
python dlc_video_overlay.py
```

### Option 2: Create Mac App with py2app
```bash
# Install py2app
pip install py2app

# Build the app
python setup.py py2app

# The app will be in dist/DLC Video Overlay.app
# Copy to Applications:
cp -r "dist/DLC Video Overlay.app" /Applications/
```

## First-Time macOS Security

If macOS blocks the app:
1. Go to **System Preferences** > **Security & Privacy**
2. Click **"Open Anyway"** to allow the app

## Usage Guide

1. **Launch** the application
2. **Browse** for the video file (MP4, AVI, MOV, MKV)
3. **Browse** for DLC file (or it auto-loads from predictions folder)
4. **Adjust playback speed** slider (0.1x - 2.0x)
   - 0.1x = super slow motion
   - 1.0x = normal speed
   - 2.0x = double speed
5. **Click Load Files**
6. **Select** which DLC points to display
7. **Play** and view the tracking data!

## Playback Controls

- **Play/Pause**: Start/stop video playback
- **Stop**: Return to first frame
- **±1 frame**: Step forward/backward one frame
- **±10 frames**: Skip forward/backward 10 frames
- **Slider**: Seek to any frame
- **Speed Slider**: Adjust playback speed in real-time

## Troubleshooting

### Dependencies Missing
```bash
pip install PyQt5 opencv-python pandas numpy pyarrow fastparquet
```

### Can't Find Python
```bash
which python
# Use the full path shown, e.g.:
/Users/bareketdamari/anaconda3/envs/PTL/bin/python dlc_video_overlay.py
```

### App Won't Build
Try the simple Python method first - it works great without needing to build an app!

