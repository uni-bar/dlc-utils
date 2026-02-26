# DLC Video Overlay Tool

A lightweight video player with DeepLabCut point overlay capabilities for visualizing tracking data on videos.

## Features

- **Video Playback**: Frame-by-frame video player with standard controls (play/pause/stop)
- **DLC Overlay**: Visualize DeepLabCut tracking points (parquet or CSV) overlaid on video
- **Point Selection**: Enable/disable individual tracking points with checkboxes
- **Custom Colors**: Assign colors to each tracking point
- **Manual Edit Mode**: Drag points on the video to correct labels and write changes back to DLC file
- **Playback Speed Control**: Adjust video speed from 0.1x to 2.0x for slow-motion or fast-forward
- **Annotation Support**: Display additional timestamp-based annotations
- **Frame Navigation**: Skip forward/backward, seek to specific frames
- **Start Time**: Begin playback from any second in the video
- **Auto-load DLC**: Automatically detects DLC files in predictions folder
- **Preferences**: Automatically saves point colors, selections, and last used files

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Ubuntu Double-Click Launcher

To create a desktop icon that launches the app with a double-click on Ubuntu:
```bash
chmod +x run_app_linux.sh create_ubuntu_launcher.sh
./create_ubuntu_launcher.sh
```

This creates:
- `~/Desktop/DLC Video Overlay.desktop`
- `~/.local/share/applications/dlc-video-overlay.desktop`

Then double-click `DLC Video Overlay` on the Desktop.

## Manual Label Editing

After loading video + DLC:
1. Click `Edit: OFF` to switch to `Edit: ON`
2. Click and drag a point to the correct location
   - edits target `*_cam_x` / `*_cam_y` points (raw camera coordinates)
3. Release mouse to:
   - update the in-memory row
   - queue autosave to the original DLC file after short idle time (same parquet/csv you loaded)
   - autosave writes DLC data only (lightweight)
   - use `Save Edits Now` to flush immediately and export corrected frames
4. Click `No Head (Clear Frame)` to clear all points for the current frame
   - sets all point `x/y` values on that frame to `NaN`
   - sets available confidence columns to `0.0`
   - exports the frame as `NO_HEAD` in the edit log
5. Click `Apply Prev -> Current` to copy all editable `_cam_` values from the previous frame

Export location:
- `<video_folder>/manual_labels/<video_stem>/images/train/*.png`
- `<video_folder>/manual_labels/<video_stem>/labels/train/*.txt`
- `<video_folder>/manual_labels/<video_stem>/edits_log.csv`

In the app UI, these are shown explicitly:
- `DLC file overwritten on save: ...`
- `Export folder: ...`

## One-Click Retrain + Re-Run (Experimental)

After exporting corrected labels:
1. In `Retrain + Re-Run (Experimental)`, choose DeepLabCut `config.yaml`.
2. (Optional) Choose PreyTouch `Arena/run_model.py`.
3. Fill PreyTouch model name (the key inside `Arena/configurations/predict_config.json`).
4. Set camera name and lightweight iterations (for fast fine-tuning).
5. Click `Retrain + Re-run This Video`.

What this workflow does:
- Converts your exported manual labels into DLC `labeled-data/.../CollectedData_<scorer>.csv/.h5`
- Runs `deeplabcut.create_training_dataset` + `deeplabcut.train_network`
- Exports latest model
- If `run_model.py` + model name are provided:
  - updates `Arena/configurations/predict_config.json` model path to the newly exported model
  - runs prediction on the currently loaded video via PreyTouch model code

Logs are written to:
- `<video_folder>/manual_labels/<video_stem>/retrain_logs/retrain_*.log`

## Usage

Run the application:
```bash
python dlc_video_overlay.py
```

### Workflow

1. **Select Video**: Click "Browse" next to Video File and select the video file (.mp4, .avi, .mov, .mkv)

2. **Select DLC Data**: Click "Browse" next to DLC Data File and select the corresponding DLC tracking file (.parquet or .csv)
   - Default location is the video's folder
   - File should contain columns like: `left_ear_x`, `left_ear_y`, `nose_x`, `nose_y`, etc.

3. **Optional Annotation File**: Select an additional file with timestamp-based annotations
   - Should contain a timestamp/frame column
   - Additional columns will be displayed as text overlays

4. **Set Start Time**: Enter the second to start playback from (default: 0)

5. **Click "Load Files"**: This will:
   - Load the video and DLC data
   - Parse tracking points from DLC columns
   - Display point checkboxes with random colors
   - Show the first frame

6. **Configure Points**:
   - Check/uncheck points to show/hide them
   - Click color buttons to change point colors
   - All selections are automatically saved

7. **Playback Controls**:
   - **Play/Pause**: Toggle video playback
   - **Stop**: Stop and return to start frame
   - **-10/-1/+1/+10**: Skip frames backward/forward
   - **Slider**: Seek to any frame (when paused)

## File Format Requirements

### DLC Files (CSV/Parquet)
- Each tracking point should have `_x` and `_y` columns
- Example: `nose_x`, `nose_y`, `left_ear_x`, `left_ear_y`
- Optional: `_likelihood` columns

### Annotation Files (CSV/Parquet/JSON)
- Must contain one of: `timestamp`, `frame`, `frame_id`, or `time` column
- Additional columns will be displayed as text annotations
- Special: `trial_id` column displays prominently in yellow

## Features in Detail

### Automatic Frame Synchronization
The tool automatically synchronizes the video frame with the corresponding DLC data row, accounting for start time offsets.

### Preference Persistence
Point colors and enabled states are saved to `~/.dlc_video_overlay_prefs.json` and restored on next launch.

### Annotation Display
- Non-coordinate annotations appear in the top-right corner
- `trial_id` annotations appear prominently in yellow
- Annotations match the current frame/timestamp

## Keyboard Shortcuts
- Spacebar: Play/Pause (when focused on play button)
- Use mouse to click frame skip buttons

## Performance
- Lightweight and efficient
- Frame-by-frame rendering with OpenCV
- Minimal memory footprint
- Smooth playback at video's native FPS

## Troubleshooting

**Video won't load**: Ensure the video file format is supported by OpenCV (MP4 with H.264 codec works best)

**DLC points not showing**: Verify column names have `_x` and `_y` suffixes

**Annotations not appearing**: Check that annotation file has a timestamp/frame column

**Slow playback**: Try reducing video resolution or lowering FPS

## Requirements
- Python 3.7+
- opencv-python
- PyQt5
- pandas
- numpy
- pyarrow (for parquet support)
- fastparquet (for parquet support)
