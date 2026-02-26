# 🚀 Creating and Updating the Desktop App

## What's Installed

A Mac `.app` file on the Desktop: **"DLC Video Overlay"** with a custom blue icon.

## How It Works

The app is a Mac application bundle with this structure:

```
DLC Video Overlay.app/
├── Contents/
    ├── Info.plist              # App metadata (name, version, etc.)
    ├── MacOS/
    │   └── launcher            # Bash script that launches Python
    └── Resources/
        └── AppIcon.icns        # The icon with tracking dots
```

**Key Point:** The app is a launcher - it runs the Python code from:
```
/Users/bareketdamari/Dev/PreyTouch/sandbox/videos/dlc_video_overlay.py
```

Changes made to the Python file are automatically used by the app.

## 🔄 Updating After Code Changes

### Most Changes: No Action Required

If only `dlc_video_overlay.py` was modified (features, fixes, etc.):
- ✅ **No need to recreate the app**
- ✅ Double-click the existing app
- ✅ It uses the latest code

### Recreating the App

Only recreate to:
- Update the icon design
- Change the app name
- Create a fresh copy

**To recreate:**
```bash
cd /Users/bareketdamari/Dev/PreyTouch/sandbox/videos
./create_mac_app.sh
```

This takes ~2 seconds and creates a fresh app on the Desktop.

## 🎨 Customizing the Icon

The icon shows:
- 🔵 Blue background (video theme)
- ⬜ White frame (video player)
- ▶️ Blue play triangle
- 🔴 4 colored dots (tracking points)

### To Change Colors/Design:

1. **Edit the icon script:**
   ```bash
   open -e /Users/bareketdamari/Dev/PreyTouch/sandbox/videos/create_mac_app.sh
   ```

2. **Find this section** (around line 67):
   ```python
   # Create a 512x512 image with a cute background
   size = 512
   img = Image.new('RGB', (size, size), color='#4A90E2')  # ← Background color
   ```

3. **Customize colors:**
   ```python
   # Background color
   color='#4A90E2'  # Current: Blue
   # Try: '#FF6B6B' (red), '#4ECDC4' (teal), '#9B59B6' (purple)
   
   # Tracking dot colors
   colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#A8E6CF']
   # Red,      Teal,     Yellow,   Green
   ```

4. **Recreate the app:**
   ```bash
   ./create_mac_app.sh
   ```

### Color Picker Tool:
Use [HTML Color Codes](https://htmlcolorcodes.com/) to pick colors.

## 📦 Sharing With Others

### Quick Share (Zip File):

```bash
cd ~/Desktop
zip -r "DLC_Video_Overlay.zip" "DLC Video Overlay.app"
```

Send them the zip file with these instructions:

**For Recipients:**
1. Unzip the file
2. Move "DLC Video Overlay.app" to Desktop or Applications
3. Right-click → "Open" (first time only for security)
4. Double-click to launch!

**⚠️ Requirements:**
- macOS 10.13 or later
- Python 3.7+ or Anaconda (with PTL environment)
- Dependencies: `pip install PyQt5 opencv-python pandas numpy pyarrow fastparquet`

### Distribution Setup:

For sharing with multiple people, include setup instructions:

```bash
# One-time setup
pip install PyQt5 opencv-python pandas numpy pyarrow fastparquet
```

## 🔧 Advanced Customization

### Change App Name:

Edit `create_mac_app.sh` line 4:
```bash
APP_NAME="DLC Video Overlay"  # ← Change this
```

### Change Where It's Created:

Edit `create_mac_app.sh` line 5:
```bash
APP_DIR="$HOME/Desktop/$APP_NAME.app"  # ← Change Desktop to anywhere
# Examples:
# "$HOME/Applications/$APP_NAME.app"
# "/Applications/$APP_NAME.app"
```

### Add Custom Icon File:

If there is a custom `.icns` icon file:

1. Replace the icon creation section (lines 60-110)
2. Copy the icon:
   ```bash
   cp /path/to/icon.icns "$RESOURCES_DIR/AppIcon.icns"
   ```

## 📋 Quick Reference

| Task | Command |
|------|---------|
| Create/recreate app | `./create_mac_app.sh` |
| Test changes | Double-click the app (no recreation needed) |
| Share with others | `zip -r ~/Desktop/DLC_Video_Overlay.zip ~/Desktop/"DLC Video Overlay.app"` |
| Customize icon | Edit `create_mac_app.sh` → run it |
| Move to Applications | `mv ~/Desktop/"DLC Video Overlay.app" /Applications/` |

## 🎯 Typical Workflow

### Day-to-Day Development:
1. Edit `dlc_video_overlay.py`
2. Double-click the Desktop app to test
3. Repeat

### Publishing an Update:
1. Make all code changes
2. Test by double-clicking the app
3. When ready, run `./create_mac_app.sh` to create fresh app
4. Zip and share

### After Major Changes:
1. Update version in `create_mac_app.sh` (line 45-46):
   ```xml
   <key>CFBundleShortVersionString</key>
   <string>1.0.0</string>  <!-- ← Change this -->
   ```
2. Recreate: `./create_mac_app.sh`

## 💡 Tips

- **The app is tiny** (~few KB) because it's just a launcher
- **All code stays in the videos folder** - easy to edit and version control
- **Icon cached by macOS** - if icon doesn't update, restart Finder: `killall Finder`
- **First launch warning** - Right-click → "Open" required on first launch

## 🐛 Troubleshooting

**App won't open:**
- Right-click → "Open" instead of double-click (first time)
- Check: Is Python/Anaconda installed?
- Check: Is PTL environment available?

**Icon doesn't show:**
- macOS caches icons - restart Finder: `killall Finder`
- Or rename the app slightly to force refresh

**Python not found:**
- Edit `create_mac_app.sh` line 13 to use correct Python path
- Or ensure PTL conda environment exists

**Want to reset everything:**
```bash
rm -rf ~/Desktop/"DLC Video Overlay.app"
./create_mac_app.sh
```

---

**Complete:** Professional-looking Mac app that's easy to update and share. 🎉
