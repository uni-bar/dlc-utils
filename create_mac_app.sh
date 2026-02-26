#!/bin/bash
# Create a Mac .app bundle for DLC Video Overlay

APP_NAME="DLC Video Overlay"
APP_DIR="$HOME/Desktop/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Creating Mac app: $APP_NAME.app on Desktop..."

# Create app bundle structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create the executable script
cat > "$MACOS_DIR/launcher" << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")/../../.."
SOURCE_DIR="/Users/bareketdamari/Dev/PreyTouch/sandbox/videos"

# Activate conda environment
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate PTL 2>/dev/null
fi

# Run the app
if [ -f "$HOME/anaconda3/envs/PTL/bin/python" ]; then
    "$HOME/anaconda3/envs/PTL/bin/python" "$SOURCE_DIR/dlc_video_overlay.py"
else
    osascript -e 'tell app "System Events" to display dialog "Python environment not found!" buttons {"OK"} default button 1'
fi
SCRIPT

chmod +x "$MACOS_DIR/launcher"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.preytouch.dlcvideooverlay</string>
    <key>CFBundleName</key>
    <string>DLC Video Overlay</string>
    <key>CFBundleDisplayName</key>
    <string>DLC Video Overlay</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
PLIST

# Create a cute icon using emoji/symbol
cat > "$RESOURCES_DIR/create_icon.py" << 'PYTHON'
#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont

# Create a 512x512 image with a cute background
size = 512
img = Image.new('RGB', (size, size), color='#4A90E2')

# Draw a simple video/play icon design
draw = ImageDraw.Draw(img)

# Draw a rounded rectangle (video frame)
rect_margin = 80
draw.rounded_rectangle(
    [rect_margin, rect_margin, size-rect_margin, size-rect_margin],
    radius=40,
    fill='#FFFFFF',
    outline='#2E5C8A',
    width=8
)

# Draw a play triangle in the center
triangle = [
    (size//2 - 60, size//2 - 80),
    (size//2 - 60, size//2 + 80),
    (size//2 + 80, size//2)
]
draw.polygon(triangle, fill='#4A90E2')

# Draw some tracking points (colored dots)
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#A8E6CF']
positions = [
    (200, 200), (320, 180), (240, 300), (340, 320)
]
for pos, color in zip(positions, colors):
    draw.ellipse([pos[0]-12, pos[1]-12, pos[0]+12, pos[1]+12], fill=color, outline='white', width=3)

# Save as PNG
output_path = os.path.join(os.path.dirname(__file__), 'AppIcon.png')
img.save(output_path, 'PNG')
print(f"Icon created: {output_path}")

# Convert to .icns format for macOS
try:
    import subprocess
    iconset_dir = output_path.replace('.png', '.iconset')
    os.makedirs(iconset_dir, exist_ok=True)
    
    # Create different sizes
    sizes = [16, 32, 64, 128, 256, 512]
    for s in sizes:
        resized = img.resize((s, s), Image.Resampling.LANCZOS)
        resized.save(f"{iconset_dir}/icon_{s}x{s}.png")
        if s <= 256:
            resized2x = img.resize((s*2, s*2), Image.Resampling.LANCZOS)
            resized2x.save(f"{iconset_dir}/icon_{s}x{s}@2x.png")
    
    # Convert to icns
    icns_path = output_path.replace('.png', '.icns')
    subprocess.run(['iconutil', '-c', 'icns', iconset_dir, '-o', icns_path])
    print(f"Icon set created: {icns_path}")
except Exception as e:
    print(f"Could not create .icns: {e}")
PYTHON

# Try to create icon with PIL
if command -v python3 &> /dev/null; then
    python3 -m pip install --quiet Pillow 2>/dev/null
    python3 "$RESOURCES_DIR/create_icon.py" 2>/dev/null
fi

# If PIL not available, create a simple colored icon using sips
if [ ! -f "$RESOURCES_DIR/AppIcon.icns" ]; then
    echo "Creating simple icon..."
    # Create a simple colored square as fallback
    cat > "$RESOURCES_DIR/icon.svg" << 'SVG'
<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
  <rect width="512" height="512" rx="80" fill="#4A90E2"/>
  <rect x="80" y="80" width="352" height="352" rx="30" fill="white" stroke="#2E5C8A" stroke-width="8"/>
  <polygon points="200,180 200,332 340,256" fill="#4A90E2"/>
  <circle cx="200" cy="200" r="12" fill="#FF6B6B" stroke="white" stroke-width="3"/>
  <circle cx="320" cy="180" r="12" fill="#4ECDC4" stroke="white" stroke-width="3"/>
  <circle cx="240" cy="300" r="12" fill="#FFE66D" stroke="white" stroke-width="3"/>
  <circle cx="340" cy="320" r="12" fill="#A8E6CF" stroke="white" stroke-width="3"/>
</svg>
SVG
fi

echo "✅ App created at: $APP_DIR"
echo "📍 You can now drag it anywhere or keep it on Desktop!"
echo ""
echo "To open: Double-click 'DLC Video Overlay' on your Desktop"
