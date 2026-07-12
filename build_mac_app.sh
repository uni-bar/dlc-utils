#!/bin/bash
set -e

# Build standalone Mac app for DLC Video Overlay Tool

echo "Building Mac application for DLC Video Overlay Tool..."

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Create the app bundle
pyinstaller --clean \
    --noconfirm \
    --name "DLC Video Overlay" \
    --windowed \
    --add-data "README.md:." \
    --add-data "retrain_dlc_from_manual_labels.py:." \
    --add-data "configs/head_only_config.yaml:configs" \
    --hidden-import=PyQt5 \
    --hidden-import=cv2 \
    --hidden-import=pandas \
    --hidden-import=numpy \
    --hidden-import=pyarrow \
    --exclude-module=matplotlib \
    --exclude-module=mpl_toolkits \
    --osx-bundle-identifier=com.preytouch.dlcvideooverlay \
    dlc_video_overlay.py

echo ""
echo "Build complete!"
echo "Application created at: dist/DLC Video Overlay.app"
echo ""
echo "To install:"
echo "  cp -r 'dist/DLC Video Overlay.app' /Applications/"
echo ""
echo "Or double-click the app in the dist folder to run it directly."
