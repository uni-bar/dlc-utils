#!/usr/bin/env bash
# Create Ubuntu desktop launchers for DLC Video Overlay.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SOURCE_DIR/run_app_linux.sh"
APP_NAME="DLC Video Overlay"
DESKTOP_FILE_NAME="$APP_NAME.desktop"
APPS_DIR="$HOME/.local/share/applications"

if [ ! -f "$LAUNCH_SCRIPT" ]; then
    echo "ERROR: Missing launcher script: $LAUNCH_SCRIPT"
    exit 1
fi

if command -v xdg-user-dir >/dev/null 2>&1; then
    DESKTOP_DIR="$(xdg-user-dir DESKTOP)"
else
    DESKTOP_DIR="$HOME/Desktop"
fi

if [ -z "${DESKTOP_DIR:-}" ]; then
    DESKTOP_DIR="$HOME/Desktop"
fi

mkdir -p "$DESKTOP_DIR"
mkdir -p "$APPS_DIR"
chmod +x "$LAUNCH_SCRIPT"

create_desktop_file() {
    local output_file="$1"
    cat > "$output_file" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$APP_NAME
Comment=Video player with DeepLabCut point overlays
Exec="$LAUNCH_SCRIPT"
Path=$SOURCE_DIR
Terminal=false
StartupNotify=true
Categories=Science;Video;
Icon=applications-multimedia
EOF
    chmod +x "$output_file"
}

DESKTOP_LAUNCHER="$DESKTOP_DIR/$DESKTOP_FILE_NAME"
APPS_LAUNCHER="$APPS_DIR/dlc-video-overlay.desktop"

create_desktop_file "$DESKTOP_LAUNCHER"
create_desktop_file "$APPS_LAUNCHER"

if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_LAUNCHER" metadata::trusted true 2>/dev/null || true
fi

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi

echo "Created Ubuntu launchers:"
echo "  Desktop: $DESKTOP_LAUNCHER"
echo "  App menu: $APPS_LAUNCHER"
echo ""
echo "You can now double-click '$APP_NAME' from your Desktop."
