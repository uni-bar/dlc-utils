#!/usr/bin/env bash
# Linux launcher for DLC Video Overlay Tool.

set -u

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SOURCE_DIR/dlc_video_overlay.py"

show_error() {
    local message="$1"
    echo "ERROR: $message" >&2
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="DLC Video Overlay" --text="$message" >/dev/null 2>&1 || true
    elif command -v xmessage >/dev/null 2>&1; then
        xmessage -center "$message" >/dev/null 2>&1 || true
    fi
}

if [ ! -f "$APP_FILE" ]; then
    show_error "Could not find dlc_video_overlay.py in: $SOURCE_DIR"
    exit 1
fi

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate PTL >/dev/null 2>&1 || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate PTL >/dev/null 2>&1 || true
fi

PYTHON_BIN=""
if [ -x "$HOME/anaconda3/envs/PTL/bin/python" ]; then
    PYTHON_BIN="$HOME/anaconda3/envs/PTL/bin/python"
elif [ -x "$HOME/miniconda3/envs/PTL/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/PTL/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

if [ -z "$PYTHON_BIN" ]; then
    show_error "Python not found. Install Python 3 or Anaconda, then try again."
    exit 1
fi

cd "$SOURCE_DIR"
"$PYTHON_BIN" "$APP_FILE"
exit_code=$?

if [ $exit_code -ne 0 ]; then
    show_error "The app exited with an error (code $exit_code). Run from terminal for details."
fi

exit $exit_code
