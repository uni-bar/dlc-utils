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

## Rigid Head Correction

`Apply rigid head` learns a trustworthy head triangle and runs a forward/backward
Kalman smoother over its camera-space center, rotation, and scale. Frames with
two trustworthy landmarks can anchor the rigid pose. The expected triangle shape
is the robust median of complete, plausible triangles within 10 frames, so normal
local perspective changes are not compared with one global shape. Every landmark
with probability below 0.5 is replaced. With two reliable current landmarks the
local rigid triangle is fitted to both; with one reliable landmark the smoothed
Kalman pose is translated to preserve that landmark. The ear midpoint is always
recalculated from the final ear positions. Reconstructed frames show an orange
`RIGID KALMAN` badge that names the replaced landmarks. A problematic frame with
no reliable current landmark or no nearby Kalman pose remains raw and shows a
yellow `KALMAN UNRESOLVED` badge.

Rigid-overlay points can be edited directly. The visible rigid position is used
for selecting and dragging, while the committed manual label is written to the
authoritative raw `*_cam` columns with confidence 1.0. The current rigid row and
ear midpoint update immediately; applying rigid head again incorporates the
manual label as trustworthy input.

The correction follows the coordinate pipeline used by the editor:
- raw video pixels: `nose_cam_x/y`, `left_ear_cam_x/y`, `right_ear_cam_x/y`
- rigid video pixels: `rigid_nose_cam_x/y`, etc.
- calibrated rigid coordinates: `rigid_nose_x/y`, etc., created by
  `Reapply Calibration To Edited DLC`

Raw prediction columns are retained. Use the `Raw` / `Rigid` selector to compare
the original and corrected video overlays.

## Fine-Tune and Run Predictions

After exporting corrected labels:
1. Choose the DeepLabCut `config.yaml` (`configs/head_only_config.yaml` is the default).
2. Choose the trained source model that generated the original parquet.
3. Choose an empty output folder for the retrained model.
4. Set the number of iterations and click `Retrain`.

Retraining does not require a video or parquet to be loaded. It uses the images
and labels in the selected shared manual-labels folder. Prediction does require
the target video to be loaded.

What this workflow does:
- Converts your exported manual labels into DLC `labeled-data/.../CollectedData_<scorer>.csv/.h5`
- Verifies that the source folder contains trained snapshot weights
- Uses the selected model's network type and snapshot during dataset creation
- Blocks generic ImageNet/base-weight downloads and verifies the generated training config
- Runs `deeplabcut.create_training_dataset` + `deeplabcut.train_network`
- Uses an 80/20 training/test split and runs `deeplabcut.evaluate_network`
- Writes the train/test pixel-error evaluation to the retraining log
- Exports and verifies the retrained model in the selected output folder

To run the model on the loaded video:
1. The prediction model defaults to the retrained output folder, but can be changed.
2. Choose the calibration folder/settings and camera.
3. Click `Run Prediction Model`.

Prediction invokes the bundled standalone `run_model.py` with the selected model,
loaded video, camera, and calibration arguments. It does not import Arena or use
`predict_config.json`.

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

**Linux `xcb` error pointing at `cv2/qt/plugins`**: remove GUI-enabled OpenCV
from the environment and use the headless wheel. The player GUI comes from PyQt5.

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install --force-reinstall "numpy<2" "opencv-python-headless<4.12" PyQt5
```

For retraining, install the full DeepLabCut training package in the Python
environment that launches the player. `deeplabcut-live` is only for inference
with an exported model and does not provide `create_training_dataset`,
`train_network`, or `evaluate_network`.

The current retrainer accepts TensorFlow `snapshot*.index` source weights and
the prediction runner uses DeepLabCut-Live. For both actions in one Python 3.10
environment, use the mutually compatible TensorFlow stack:

```bash
python -m pip install "numpy==1.26.4" "tensorflow==2.10.0"
python -m pip install "deeplabcut[tf]==2.3.11" "deeplabcut-live[tf]==1.1.0"
```

On a Linux GPU server, TensorFlow 2.10 also needs CUDA 11.2 and cuDNN 8.1.
Install them inside the same Conda environment and expose its library folder:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 "libstdcxx-ng>=11" "libgcc-ng>=11" pyzmq
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
conda deactivate
conda activate pt
```

Do not combine `deeplabcut[tf] 3.0.0` and `deeplabcut-live[tf] 1.1.0`: their
TensorFlow requirements do not overlap.

## Requirements
- Python 3.7+
- opencv-python-headless (PyQt5 provides the GUI; avoid OpenCV's conflicting Qt plugins)
- PyQt5
- pandas
- numpy
- pyarrow (for parquet support)
- fastparquet (for parquet support)
