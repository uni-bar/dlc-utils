# DLC Video Overlay - Quick Start

## ✅ All Changes Complete!

### What's New

1. **❌ Removed "Flip Y" checkbox** - No longer needed! Coordinates work correctly as-is.

2. **✅ Added Playback Speed Control** - New slider in the left panel
   - Range: 0.1x (super slow) to 2.0x (fast)
   - Default: 1.0x (normal speed)
   - Adjust anytime, even while playing!

## 🚀 Three Ways to Run

### Method 1: Double-Click Launcher (Easiest)
1. Navigate to `/Users/bareketdamari/Dev/PreyTouch/sandbox/videos/`
2. Double-click `run_app.command`
3. The app will open

### Method 2: Run from Terminal
```bash
cd /Users/bareketdamari/Dev/PreyTouch/sandbox/videos
python dlc_video_overlay.py
```

### Method 3: Create Standalone Mac App
```bash
cd /Users/bareketdamari/Dev/PreyTouch/sandbox/videos
pip install py2app
python setup.py py2app
cp -r "dist/DLC Video Overlay.app" /Applications/
```

## 🎮 Usage

1. **Launch** the app (any method above)
2. **Browse** for video file
3. **Browse** for DLC file (or auto-loads from predictions/)
4. **Set playback speed** with the slider
   - Try 0.5x for slow motion
   - Try 1.5x to speed through
5. **Load Files** button
6. **Select points** to display with checkboxes
7. **Play**

## 🎚️ Playback Controls

- **Speed Slider**: NEW! Control how fast video plays (0.1x - 2.0x)
- **Play/Pause**: Toggle playback
- **Stop**: Return to start
- **±1**: Step one frame
- **±10**: Skip 10 frames
- **Slider**: Seek to any frame

## 💡 Tips

- **Slow motion analysis**: Set speed to 0.2x to catch details
- **Quick review**: Set speed to 1.5x or 2.0x
- **Frame-by-frame**: Use ±1 buttons for precise control
- **Speed changes live**: Adjust speed slider while playing

## 🔧 Need Help?

If Python not found:
```bash
which python3
# Then use full path:
/path/to/python3 dlc_video_overlay.py
```

Missing packages:
```bash
pip install PyQt5 "opencv-python-headless<4.12" pandas "numpy<2" pyarrow fastparquet
```

## 📍 File Locations

- Main app: `/Users/bareketdamari/Dev/PreyTouch/sandbox/videos/dlc_video_overlay.py`
- Launcher: `/Users/bareketdamari/Dev/PreyTouch/sandbox/videos/run_app.command`
- Full guide: `/Users/bareketdamari/Dev/PreyTouch/sandbox/videos/INSTALL_MAC.md`

---

**Ready to go:** Double-click `run_app.command` or run `python dlc_video_overlay.py` 🎉
