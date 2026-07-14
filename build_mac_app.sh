#!/bin/bash
set -e

# Build standalone Mac app for DLC Video Overlay Tool

echo "Building Mac application for DLC Video Overlay Tool..."

VENV_DIR=".venv-mac-build"
PYTHON="$VENV_DIR/bin/python"
PYINSTALLER="$VENV_DIR/bin/pyinstaller"

if [ ! -x "$PYTHON" ]; then
    python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON" -c 'import PyInstaller, PyQt5, cv2, pandas, numpy, pyarrow; assert int(numpy.__version__.split(".")[0]) < 2' 2>/dev/null; then
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install pyinstaller PyQt5 "opencv-python-headless<4.12" pandas "numpy<2" pyarrow
fi

# Create the app bundle
"$PYINSTALLER" --clean \
    --noconfirm \
    --name "DLC Video Overlay" \
    --windowed \
    --add-data "README.md:." \
    --add-data "retrain_dlc_from_manual_labels.py:." \
    --add-data "run_model.py:." \
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
