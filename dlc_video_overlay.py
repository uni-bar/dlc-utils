#!/usr/bin/env python3
"""
DLC Video Overlay Tool
A lightweight video player with DeepLabCut point overlay capabilities
"""

import sys
import csv
import importlib
import json
import math
import random
import shutil
import subprocess
import traceback
from hashlib import sha1
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import cv2
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox,
    QScrollArea, QGroupBox, QSpinBox, QColorDialog, QSlider, QSizePolicy,
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QDesktopServices

DEFAULT_APP_CALIBRATION_DIR = None
run_edited_pose_calibration_job = None
CALIBRATION_HELPER_IMPORT_ERROR = None

CALIBRATION_PRESETS = {
    "reptilearn5": {
        "label": "Reptilearn 5 defaults",
        "calibration_dir": "/media/sil3/Data/Bareket/arena_configs/reptilearn5/calibrations",
        "screen_start_x": "7.59",
        "screen_pix_cm": "0.027604",
        "screen_y": "-4.3",
    },
    "reptilearn4": {
        "label": "Reptilearn 4 defaults",
        "calibration_dir": "/media/sil3/Data/Bareket/arenas_configs/reptilearn4/calibrations",
        "screen_start_x": "5.59",
        "screen_pix_cm": "0.027604",
        "screen_y": "0.06",
    },
}


def load_calibration_helper():
    """Load calibration helper only when the user actually runs calibration."""
    global DEFAULT_APP_CALIBRATION_DIR, run_edited_pose_calibration_job, CALIBRATION_HELPER_IMPORT_ERROR

    if run_edited_pose_calibration_job is not None:
        return run_edited_pose_calibration_job

    try:
        module = importlib.import_module("run_edited_pose_calibration")
        run_edited_pose_calibration_job = module.run_edited_pose_calibration_job
        DEFAULT_APP_CALIBRATION_DIR = getattr(module, "DEFAULT_APP_CALIBRATION_DIR", None)
        CALIBRATION_HELPER_IMPORT_ERROR = None
        return run_edited_pose_calibration_job
    except (Exception, SystemExit) as exc:
        run_edited_pose_calibration_job = None
        CALIBRATION_HELPER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None


class InteractiveVideoLabel(QLabel):
    """QLabel that exposes mouse events for interactive point editing."""

    mouse_pressed = pyqtSignal(object)
    mouse_moved = pyqtSignal(object)
    mouse_released = pyqtSignal(object)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._overlay_pos = None  # (x, y) in label coords
        self._overlay_color = (255, 220, 0)
    
    def set_drag_overlay(self, pos: Optional[Tuple[int, int]], color: Optional[Tuple[int, int, int]] = None):
        """Draw a lightweight drag marker on top of existing pixmap."""
        self._overlay_pos = pos
        if color is not None:
            self._overlay_color = color
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._overlay_pos is None:
            return
        x, y = self._overlay_pos
        r, g, b = self._overlay_color
        painter = QPainter(self)
        pen = QPen(QColor(255, 220, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(x - 10, y - 10, 20, 20)
        pen2 = QPen(QColor(r, g, b))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.drawEllipse(x - 6, y - 6, 12, 12)
        painter.end()

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)


class CopyFrameRangeDialog(QDialog):
    """Prompt for a source frame and a till frame for bulk point copying."""

    def __init__(self, min_frame: int, max_frame: int, source_frame: int, till_frame: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Points Across Frames")

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Copy editable point values from the source frame to every frame until the till frame. "
            "The source frame itself is not overwritten."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form_layout = QFormLayout()

        self.source_frame_spin = QSpinBox()
        self.source_frame_spin.setRange(min_frame, max_frame)
        self.source_frame_spin.setValue(max(min_frame, min(max_frame, source_frame)))
        form_layout.addRow("Source frame:", self.source_frame_spin)

        self.till_frame_spin = QSpinBox()
        self.till_frame_spin.setRange(min_frame, max_frame)
        self.till_frame_spin.setValue(max(min_frame, min(max_frame, till_frame)))
        form_layout.addRow("Till frame:", self.till_frame_spin)

        layout.addLayout(form_layout)

        hint_label = QLabel("Forward and backward ranges are both supported.")
        hint_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        layout.addWidget(hint_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_frames(self) -> Tuple[int, int]:
        """Return source and till frames selected in the dialog."""
        return int(self.source_frame_spin.value()), int(self.till_frame_spin.value())


class FrameRangeDialog(QDialog):
    """Prompt for an inclusive frame range."""

    def __init__(
        self,
        title: str,
        description: str,
        start_label: str,
        end_label: str,
        min_frame: int,
        max_frame: int,
        start_frame: int,
        end_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)

        info_label = QLabel(description)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form_layout = QFormLayout()

        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(min_frame, max_frame)
        self.start_frame_spin.setValue(max(min_frame, min(max_frame, start_frame)))
        form_layout.addRow(start_label, self.start_frame_spin)

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(min_frame, max_frame)
        self.end_frame_spin.setValue(max(min_frame, min(max_frame, end_frame)))
        form_layout.addRow(end_label, self.end_frame_spin)

        layout.addLayout(form_layout)

        hint_label = QLabel("The range is inclusive. Forward and backward ranges are both supported.")
        hint_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        layout.addWidget(hint_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_frames(self) -> Tuple[int, int]:
        """Return inclusive start/end frames selected in the dialog."""
        return int(self.start_frame_spin.value()), int(self.end_frame_spin.value())


def point_raw_to_pixels_for_export(
    x_raw: float, y_raw: float, frame_w: int, frame_h: int, coords_are_normalized: bool
) -> Tuple[int, int]:
    """Convert stored DLC coordinates into frame pixels for export labels."""
    if coords_are_normalized:
        x_px = int(float(x_raw) * frame_w)
        y_px = int(float(y_raw) * frame_h)
    else:
        x_px = int(float(x_raw))
        y_px = int(float(y_raw))
    x_px = max(0, min(frame_w - 1, x_px))
    y_px = max(0, min(frame_h - 1, y_px))
    return x_px, y_px


def get_snapshot_dlc_row_for_frame(
    video_frame: int,
    dlc_frame_map: Optional[Dict[int, int]],
    start_frame: int,
    row_count: int,
) -> Optional[int]:
    """Resolve a video frame to a DLC row using a snapshot of the frame map."""
    if dlc_frame_map is not None and video_frame in dlc_frame_map:
        return int(dlc_frame_map[video_frame])

    row_from_start = video_frame - start_frame
    if 0 <= row_from_start < row_count:
        return int(row_from_start)
    if 0 <= video_frame < row_count:
        return int(video_frame)
    return None


def ensure_export_log_header(export_log_path: Path):
    """Create the export log with header if it does not already exist."""
    if export_log_path.exists():
        return
    export_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "video_path",
            "dlc_path",
            "frame",
            "dlc_row",
            "edited_point",
            "image_path",
            "label_path",
        ])


class SaveExportWorker(QObject):
    """Write DLC data and optional frame/label exports off the UI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job: Dict[str, Any]):
        super().__init__()
        self.job = job

    def run(self):
        try:
            self.finished.emit(self._run_job())
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _read_frame_bgr(self, cap, frame_num: int) -> Optional[np.ndarray]:
        if cap is None or not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def _get_point_confidence_value(self, row_data, point_name: str) -> Optional[float]:
        candidates = [point_name]
        if point_name.endswith("_cam"):
            candidates.append(point_name[:-4])
        for candidate in candidates:
            for suffix in ("_prob", "_likelihood", "_conf"):
                conf_col = f"{candidate}{suffix}"
                if conf_col not in row_data.index:
                    continue
                conf_val = row_data.get(conf_col, np.nan)
                if pd.notna(conf_val):
                    try:
                        return float(conf_val)
                    except Exception:
                        continue
        return None

    def _write_export_label(
        self,
        dlc_row_idx: int,
        edited_point: str,
        frame_num: int,
        frame_bgr: np.ndarray,
    ):
        job = self.job
        df = job["dlc_data"]
        export_frames_dir = Path(job["export_frames_dir"])
        export_labels_dir = Path(job["export_labels_dir"])
        export_log_path = Path(job["export_log_path"])
        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return

        base_name = f"{job['video_export_prefix']}_f{int(frame_num):07d}"
        img_path = export_frames_dir / f"{base_name}.png"
        label_path = export_labels_dir / f"{base_name}.txt"

        cv2.imwrite(str(img_path), frame_bgr)

        row = df.iloc[int(dlc_row_idx)]
        lines = [
            f"# video_path={job['video_path']}",
            f"# dlc_path={job['dlc_path']}",
            f"# frame={frame_num}",
            f"# dlc_row={dlc_row_idx}",
            f"# edited_point={edited_point}",
            f"# timestamp={datetime.utcnow().isoformat()}Z",
            "point_name,x_raw,y_raw,x_pixel,y_pixel,x_norm,y_norm,confidence",
        ]

        for point_name in job["editable_point_names"]:
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col not in df.columns or y_col not in df.columns:
                continue
            x_val = row[x_col]
            y_val = row[y_col]
            if pd.isna(x_val) or pd.isna(y_val):
                continue

            x_raw = float(x_val)
            y_raw = float(y_val)
            x_px, y_px = point_raw_to_pixels_for_export(
                x_raw, y_raw, frame_w, frame_h, bool(job["coords_are_normalized"])
            )
            x_norm = x_raw if job["coords_are_normalized"] else (x_raw / max(frame_w, 1))
            y_norm = y_raw if job["coords_are_normalized"] else (y_raw / max(frame_h, 1))
            conf_val = self._get_point_confidence_value(row, point_name)
            conf_text = "" if conf_val is None else f"{conf_val:.6f}"
            lines.append(
                f"{point_name},{x_raw:.6f},{y_raw:.6f},{x_px},{y_px},{x_norm:.6f},{y_norm:.6f},{conf_text}"
            )

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        with open(export_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                job["video_path"],
                job["dlc_path"],
                frame_num,
                dlc_row_idx,
                edited_point,
                str(img_path),
                str(label_path),
            ])

    def _run_job(self) -> Dict[str, Any]:
        job = self.job
        dlc_file = Path(job["dlc_path"])
        is_parquet = bool(job["is_parquet"])
        tmp_file = (
            dlc_file.with_name(dlc_file.name + ".tmp.parquet")
            if is_parquet else
            dlc_file.with_name(dlc_file.name + ".tmp.csv")
        )
        backup_file = dlc_file.with_suffix(dlc_file.suffix + ".bak")
        created_backup = False

        if job["create_backup"] and dlc_file.exists() and not backup_file.exists():
            shutil.copyfile(dlc_file, backup_file)
            created_backup = True

        if is_parquet:
            job["dlc_data"].to_parquet(tmp_file, index=False)
        else:
            job["dlc_data"].to_csv(tmp_file, index=False)
        tmp_file.replace(dlc_file)

        processed_frames = []
        exported_count = 0
        pending = sorted(job["pending_export_frames"].items(), key=lambda item: item[0])
        if job["export_frames"] and pending:
            export_frames_dir = Path(job["export_frames_dir"]) if job["export_frames_dir"] else None
            export_labels_dir = Path(job["export_labels_dir"]) if job["export_labels_dir"] else None
            export_log_path = Path(job["export_log_path"]) if job["export_log_path"] else None
            if export_frames_dir is not None and export_labels_dir is not None and export_log_path is not None:
                export_frames_dir.mkdir(parents=True, exist_ok=True)
                export_labels_dir.mkdir(parents=True, exist_ok=True)
                ensure_export_log_header(export_log_path)

                cap = None
                if job["video_path"]:
                    cap = cv2.VideoCapture(str(job["video_path"]))
                try:
                    for frame_num, edited_points in pending:
                        frame_num = int(frame_num)
                        dlc_row_idx = get_snapshot_dlc_row_for_frame(
                            video_frame=frame_num,
                            dlc_frame_map=job["dlc_frame_map"],
                            start_frame=int(job["start_frame"]),
                            row_count=len(job["dlc_data"]),
                        )
                        processed_frames.append(frame_num)
                        if dlc_row_idx is None:
                            continue

                        if frame_num == int(job["current_frame"]) and job["current_frame_bgr"] is not None:
                            frame_bgr = job["current_frame_bgr"].copy()
                        else:
                            frame_bgr = self._read_frame_bgr(cap, frame_num)
                        if frame_bgr is None:
                            continue

                        edited_point_text = "|".join(sorted(edited_points)) if edited_points else "EDITED"
                        self._write_export_label(
                            dlc_row_idx=dlc_row_idx,
                            edited_point=edited_point_text,
                            frame_num=frame_num,
                            frame_bgr=frame_bgr,
                        )
                        exported_count += 1
                finally:
                    if cap is not None:
                        cap.release()

        return {
            "dlc_path": str(dlc_file),
            "revision": int(job["revision"]),
            "export_frames": bool(job["export_frames"]),
            "autosave": bool(job["autosave"]),
            "processed_frames": processed_frames,
            "exported_count": exported_count,
            "export_root": job["export_root"],
            "created_backup": created_backup,
        }


class CalibrationWorker(QObject):
    """Reapply calibrated pose columns off the UI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job: Dict[str, Any]):
        super().__init__()
        self.job = job

    def run(self):
        try:
            helper = load_calibration_helper()
            if helper is None:
                raise RuntimeError(
                    "Calibration helper could not be imported"
                    + (f": {CALIBRATION_HELPER_IMPORT_ERROR}" if CALIBRATION_HELPER_IMPORT_ERROR else "")
                )

            log_lines: List[str] = []
            result = helper(
                pose_file=self.job["pose_file"],
                video_path=self.job["video_path"],
                calibration_dir=self.job["calibration_dir"],
                screen_start_x=self.job.get("screen_start_x"),
                screen_pix_cm=self.job.get("screen_pix_cm"),
                screen_y=self.job.get("screen_y"),
                logger=lambda message: log_lines.append(str(message)),
            )
            result["log_lines"] = log_lines
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class VideoOverlayPlayer(QMainWindow):
    """Main application window for video overlay player"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Video Overlay Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # State variables
        self.video_path = None
        self.dlc_path = None
        self.annotation_path = None
        self.cap = None
        self.dlc_data = None
        self.annotation_data = None
        self.current_frame = 0
        self.start_second = 0
        self.start_frame = 0
        self.is_playing = False
        self.fps = 30
        self.total_frames = 0
        self.playback_speed = 1.0
        
        # Point configuration
        self.point_configs = {}  # {point_name: {'enabled': bool, 'color': (r,g,b)}}
        self.prefs_file = Path.home() / ".dlc_video_overlay_prefs.json"
        
        # Frame mapping
        self.dlc_frame_map = None  # Maps video frame to DLC row index
        self._current_trial_id = None  # Current trial ID from DLC data
        
        # Remember last browse locations
        self.last_video_dir = str(Path.home())
        self.last_dlc_dir = str(Path.home())
        
        # Coordinate system configuration
        # DLC uses standard OpenCV coordinates (origin at top-left)
        self.coords_are_normalized = False  # Will be auto-detected
        
        # Edit mode state
        self.edit_mode = False
        self.dragging_point_name = None
        self.dragging_dlc_row_idx = None
        self.drag_start_frame = None
        self.drag_start_frame_pos = None  # (x_px, y_px) anchor at drag start
        self.drag_all_points = False
        self.drag_rotate_points = False
        self.drag_source_points_raw = {}  # {point_name: (x_raw, y_raw)} captured at drag start
        self.drag_rotation_center_point = None
        self.drag_rotation_center_px = None
        self.drag_rotation_start_angle = None
        self.drag_radius_px = 24
        self._last_rendered_frame_size = None  # (width, height)
        self._last_rendered_bgr = None
        self._display_rect = None  # (x_offset, y_offset, width, height)
        self.editable_point_names = set()
        self.drag_preview_raw = None  # (x_raw, y_raw) used only while dragging
        self.drag_preview_points_raw = {}  # {point_name: (x_raw, y_raw)} for shift-drag preview
        self._last_rendered_rgb_base = None
        self.pending_export_frames = {}  # {frame_num: set([point_name, ...])}
        self.pending_export_frame_revisions = {}
        self._pending_save = False
        self._dlc_backup_created = False
        self._edit_revision = 0
        self._save_in_progress = False
        self._save_followup_requested = False
        self._save_followup_export = False
        self._save_thread = None
        self._save_worker = None
        self._load_after_save = False
        self._load_after_save_paths = None
        self._close_after_save = False
        self._retrain_after_save = False
        self._calibration_after_save = False
        self._close_after_calibration = False
        self._loaded_dlc_signature = None
        self.export_root = None
        self.export_frames_dir = None
        self.export_labels_dir = None
        self.export_log_path = None
        self.draw_point_names = False
        self.retrain_process = None
        self.retrain_log_handle = None
        self.retrain_log_path = None
        self.retrain_helper_script = Path(__file__).with_name("retrain_dlc_from_manual_labels.py")
        self._calibration_in_progress = False
        self._calibration_thread = None
        self._calibration_worker = None
        self.default_calibration_dir = (
            Path(DEFAULT_APP_CALIBRATION_DIR)
            if DEFAULT_APP_CALIBRATION_DIR else None
        )
        # Default shared manual-labels root in the source/app folder if user does not choose one.
        self.default_manual_labels_root = Path(__file__).resolve().parent / "manual_labels"
        self._flattened_manual_labels_roots = set()
        self.manual_edit_confidence = 1.0
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # Debounced save to keep drag edits responsive
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.autosave_edits)
        
        # Poll background retrain process without blocking UI
        self.retrain_poll_timer = QTimer()
        self.retrain_poll_timer.timeout.connect(self.poll_retrain_process)
        
        self.init_ui()
        self.load_preferences()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(420)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # Video file selection
        video_group = QGroupBox("Video File")
        video_layout = QVBoxLayout()
        video_input_layout = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Select video file...")
        video_browse_btn = QPushButton("Browse")
        video_browse_btn.clicked.connect(self.browse_video)
        video_input_layout.addWidget(self.video_input)
        video_input_layout.addWidget(video_browse_btn)
        video_layout.addLayout(video_input_layout)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)
        
        # DLC file selection
        dlc_group = QGroupBox("DLC Data File")
        dlc_layout = QVBoxLayout()
        dlc_input_layout = QHBoxLayout()
        self.dlc_input = QLineEdit()
        self.dlc_input.setPlaceholderText("Select DLC parquet/csv file...")
        dlc_browse_btn = QPushButton("Browse")
        dlc_browse_btn.clicked.connect(self.browse_dlc)
        dlc_input_layout.addWidget(self.dlc_input)
        dlc_input_layout.addWidget(dlc_browse_btn)
        dlc_layout.addLayout(dlc_input_layout)
        dlc_group.setLayout(dlc_layout)
        left_layout.addWidget(dlc_group)
        
        # Annotation file selection
        annotation_group = QGroupBox("Annotation File (Optional)")
        annotation_layout = QVBoxLayout()
        annotation_input_layout = QHBoxLayout()
        self.annotation_input = QLineEdit()
        self.annotation_input.setPlaceholderText("Select annotation file...")
        annotation_browse_btn = QPushButton("Browse")
        annotation_browse_btn.clicked.connect(self.browse_annotation)
        annotation_input_layout.addWidget(self.annotation_input)
        annotation_input_layout.addWidget(annotation_browse_btn)
        annotation_layout.addLayout(annotation_input_layout)
        annotation_group.setLayout(annotation_layout)
        left_layout.addWidget(annotation_group)
        
        # Start second input
        start_group = QGroupBox("Start Time")
        start_layout = QHBoxLayout()
        self.start_second_spin = QSpinBox()
        self.start_second_spin.setMinimum(0)
        self.start_second_spin.setMaximum(999999)
        self.start_second_spin.setSuffix(" seconds")
        self.start_second_spin.valueChanged.connect(self.on_start_second_changed)
        start_layout.addWidget(QLabel("Start from:"))
        start_layout.addWidget(self.start_second_spin)
        start_group.setLayout(start_layout)
        left_layout.addWidget(start_group)
        
        # Playback speed control
        speed_group = QGroupBox("Playback Speed")
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(50)
        speed_layout.addWidget(QLabel("Speed:"))
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        speed_group.setLayout(speed_layout)
        left_layout.addWidget(speed_group)
        
        # Load button
        load_btn = QPushButton("Load Files")
        load_btn.clicked.connect(self.load_files)
        load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        left_layout.addWidget(load_btn)
        
        # Manual edit controls
        edit_group = QGroupBox("Manual Edit")
        edit_layout = QVBoxLayout()
        
        self.edit_mode_btn = QPushButton("Edit: OFF")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.clicked.connect(self.toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_btn)
        
        self.save_edits_btn = QPushButton("Save Edits Now")
        self.save_edits_btn.clicked.connect(self.persist_dlc_edits)
        edit_layout.addWidget(self.save_edits_btn)
        
        self.clear_frame_btn = QPushButton("No Head (Clear Frame)")
        self.clear_frame_btn.setStyleSheet(
            "QPushButton { background-color: #B71C1C; color: white; font-weight: bold; }"
        )
        self.clear_frame_btn.clicked.connect(self.clear_current_frame_points)
        edit_layout.addWidget(self.clear_frame_btn)
        
        self.apply_prev_btn = QPushButton("Apply Prev -> Current")
        self.apply_prev_btn.clicked.connect(self.apply_previous_frame_to_current)
        edit_layout.addWidget(self.apply_prev_btn)

        self.apply_next10_btn = QPushButton("Apply Current -> Next 10")
        self.apply_next10_btn.clicked.connect(self.apply_current_frame_to_next_10)
        edit_layout.addWidget(self.apply_next10_btn)

        self.apply_source_range_btn = QPushButton("Apply Source -> Till...")
        self.apply_source_range_btn.clicked.connect(self.prompt_copy_source_frame_range)
        edit_layout.addWidget(self.apply_source_range_btn)

        self.flip_ears_btn = QPushButton("Flip L/R Ears")
        self.flip_ears_btn.clicked.connect(self.flip_ear_points_current_frame)
        self.flip_ears_btn.setEnabled(False)
        edit_layout.addWidget(self.flip_ears_btn)

        self.flip_ears_range_btn = QPushButton("Flip L/R Ears A -> B")
        self.flip_ears_range_btn.clicked.connect(self.prompt_flip_ears_frame_range)
        self.flip_ears_range_btn.setEnabled(False)
        edit_layout.addWidget(self.flip_ears_range_btn)

        clear_point_row = QHBoxLayout()
        self.clear_point_combo = QComboBox()
        self.clear_point_combo.setPlaceholderText("Select point...")
        self.clear_point_btn = QPushButton("Clear Point")
        self.clear_point_btn.clicked.connect(self.clear_selected_point_current_frame)
        clear_point_row.addWidget(self.clear_point_combo, 1)
        clear_point_row.addWidget(self.clear_point_btn)
        edit_layout.addLayout(clear_point_row)

        manual_labels_row = QHBoxLayout()
        self.manual_labels_root_input = QLineEdit(str(self.default_manual_labels_root))
        self.manual_labels_root_input.setPlaceholderText("Shared manual labels root folder")
        self.manual_labels_root_input.editingFinished.connect(self.on_manual_labels_root_changed)
        manual_labels_browse_btn = QPushButton("Browse")
        manual_labels_browse_btn.clicked.connect(self.browse_manual_labels_root)
        manual_labels_row.addWidget(self.manual_labels_root_input)
        manual_labels_row.addWidget(manual_labels_browse_btn)
        edit_layout.addLayout(manual_labels_row)
        
        self.edit_status_label = QLabel("Tip: turn Edit ON, drag a point, or hold Shift to drag all points together.")
        self.edit_status_label.setWordWrap(True)
        self.edit_status_label.setStyleSheet("font-size: 9pt; color: #444444;")
        edit_layout.addWidget(self.edit_status_label)
        
        self.export_path_label = QLabel("Export folder: (load files)")
        self.export_path_label.setWordWrap(True)
        self.export_path_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        edit_layout.addWidget(self.export_path_label)
        
        self.dlc_save_path_label = QLabel("DLC file overwritten on save: (load files)")
        self.dlc_save_path_label.setWordWrap(True)
        self.dlc_save_path_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        edit_layout.addWidget(self.dlc_save_path_label)
        
        export_actions_layout = QHBoxLayout()
        self.open_export_btn = QPushButton("Open Export Folder")
        self.open_export_btn.clicked.connect(self.open_export_folder)
        export_actions_layout.addWidget(self.open_export_btn)
        self.open_dlc_btn = QPushButton("Open DLC Folder")
        self.open_dlc_btn.clicked.connect(self.open_dlc_folder)
        export_actions_layout.addWidget(self.open_dlc_btn)
        edit_layout.addLayout(export_actions_layout)
        
        retrain_group = QGroupBox("Retrain + Re-Run (Experimental)")
        retrain_layout = QVBoxLayout()
        
        self.default_model_root = Path("/data/PreyTouch/output/models")

        retrain_cfg_row = QHBoxLayout()
        self.retrain_config_input = QLineEdit()
        self.retrain_config_input.setPlaceholderText("DeepLabCut project config.yaml")
        retrain_cfg_browse = QPushButton("Browse")
        retrain_cfg_browse.clicked.connect(self.browse_retrain_config)
        retrain_cfg_row.addWidget(self.retrain_config_input)
        retrain_cfg_row.addWidget(retrain_cfg_browse)
        retrain_layout.addLayout(retrain_cfg_row)
        
        run_model_row = QHBoxLayout()
        self.run_model_script_input = QLineEdit()
        self.run_model_script_input.setPlaceholderText("PreyTouch Arena/run_model.py (optional for re-run)")
        run_model_browse = QPushButton("Browse")
        run_model_browse.clicked.connect(self.browse_run_model_script)
        run_model_row.addWidget(self.run_model_script_input)
        run_model_row.addWidget(run_model_browse)
        retrain_layout.addLayout(run_model_row)

        model_path_row = QHBoxLayout()
        self.retrain_model_path_input = QLineEdit(str(self.default_model_root))
        self.retrain_model_path_input.setPlaceholderText("Model folder (optional)")
        model_path_browse = QPushButton("Browse")
        model_path_browse.clicked.connect(self.browse_retrain_model_path)
        model_path_row.addWidget(self.retrain_model_path_input)
        model_path_row.addWidget(model_path_browse)
        retrain_layout.addLayout(model_path_row)

        calibration_preset_row = QHBoxLayout()
        self.calibration_preset_combo = QComboBox()
        self.calibration_preset_combo.addItem("Custom/manual", "")
        for preset_key, preset in CALIBRATION_PRESETS.items():
            self.calibration_preset_combo.addItem(preset["label"], preset_key)
        self.calibration_preset_combo.currentIndexChanged.connect(self.on_calibration_preset_changed)
        calibration_preset_row.addWidget(QLabel("Preset:"))
        calibration_preset_row.addWidget(self.calibration_preset_combo)
        retrain_layout.addLayout(calibration_preset_row)

        calibration_dir_row = QHBoxLayout()
        default_calibration_text = str(self.default_calibration_dir) if self.default_calibration_dir is not None else ""
        self.calibration_dir_input = QLineEdit(default_calibration_text)
        self.calibration_dir_input.setPlaceholderText("Calibration folder")
        self.calibration_dir_input.editingFinished.connect(self.on_calibration_field_edited)
        calibration_dir_browse = QPushButton("Browse")
        calibration_dir_browse.clicked.connect(self.browse_calibration_dir)
        calibration_dir_row.addWidget(self.calibration_dir_input)
        calibration_dir_row.addWidget(calibration_dir_browse)
        retrain_layout.addLayout(calibration_dir_row)

        screen_calibration_row = QHBoxLayout()
        self.screen_start_x_input = QLineEdit("7.59")
        self.screen_start_x_input.setMaximumWidth(80)
        self.screen_start_x_input.setPlaceholderText("start_x")
        self.screen_start_x_input.editingFinished.connect(self.on_calibration_field_edited)
        self.screen_pix_cm_input = QLineEdit("0.027604")
        self.screen_pix_cm_input.setMaximumWidth(90)
        self.screen_pix_cm_input.setPlaceholderText("pix_cm")
        self.screen_pix_cm_input.editingFinished.connect(self.on_calibration_field_edited)
        self.screen_y_input = QLineEdit("-4.3")
        self.screen_y_input.setMaximumWidth(80)
        self.screen_y_input.setPlaceholderText("screen_y")
        self.screen_y_input.editingFinished.connect(self.on_calibration_field_edited)
        screen_calibration_row.addWidget(QLabel("Screen:"))
        screen_calibration_row.addWidget(self.screen_start_x_input)
        screen_calibration_row.addWidget(self.screen_pix_cm_input)
        screen_calibration_row.addWidget(self.screen_y_input)
        retrain_layout.addLayout(screen_calibration_row)
        
        retrain_args_row = QHBoxLayout()
        self.retrain_model_name_input = QLineEdit()
        self.retrain_model_name_input.setPlaceholderText("Predict model key (optional)")
        retrain_args_row.addWidget(self.retrain_model_name_input)
        self.retrain_cam_name_input = QLineEdit("top")
        self.retrain_cam_name_input.setMaximumWidth(80)
        retrain_args_row.addWidget(QLabel("Cam:"))
        retrain_args_row.addWidget(self.retrain_cam_name_input)
        self.retrain_iters_spin = QSpinBox()
        self.retrain_iters_spin.setRange(100, 1000000)
        self.retrain_iters_spin.setSingleStep(500)
        self.retrain_iters_spin.setValue(5000)
        retrain_args_row.addWidget(QLabel("Iters:"))
        retrain_args_row.addWidget(self.retrain_iters_spin)
        retrain_layout.addLayout(retrain_args_row)

        self.reapply_calibration_btn = QPushButton("Reapply Calibration To Edited DLC")
        self.reapply_calibration_btn.clicked.connect(self.start_reapply_calibration)
        self.reapply_calibration_btn.setToolTip(
            "Uses the current DLC file, current video, and the selected calibration folder."
        )
        retrain_layout.addWidget(self.reapply_calibration_btn)

        self.calibration_status_label = QLabel("Calibration status: idle")
        self.calibration_status_label.setWordWrap(True)
        self.calibration_status_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        retrain_layout.addWidget(self.calibration_status_label)
        
        self.retrain_btn = QPushButton("Retrain + Re-run This Video")
        self.retrain_btn.clicked.connect(self.start_retrain_and_rerun)
        retrain_layout.addWidget(self.retrain_btn)
        
        self.retrain_status_label = QLabel("Retrain status: idle")
        self.retrain_status_label.setWordWrap(True)
        self.retrain_status_label.setStyleSheet("font-size: 8.5pt; color: #555555;")
        retrain_layout.addWidget(self.retrain_status_label)
        
        retrain_group.setLayout(retrain_layout)
        edit_layout.addWidget(retrain_group)
        self.refresh_background_action_buttons()
        
        edit_group.setLayout(edit_layout)
        left_layout.addWidget(edit_group)
        
        # Point selection area
        points_group = QGroupBox("DLC Points")
        points_layout = QVBoxLayout()
        self.points_scroll = QScrollArea()
        self.points_scroll.setWidgetResizable(True)
        self.points_widget = QWidget()
        self.points_layout = QVBoxLayout(self.points_widget)
        self.points_scroll.setWidget(self.points_widget)
        points_layout.addWidget(self.points_scroll)
        
        # Add coordinates display below the checkboxes
        self.coords_label = QLabel("Point Coordinates:")
        self.coords_label.setWordWrap(True)
        self.coords_label.setStyleSheet("font-size: 9pt; padding: 5px; background-color: #f0f0f0; color: #000000;")
        points_layout.addWidget(self.coords_label)
        
        points_group.setLayout(points_layout)
        left_layout.addWidget(points_group)
        
        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        
        # Left panel - Video display and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Frame info at top
        self.frame_info_label = QLabel("Load video and DLC files to begin")
        self.frame_info_label.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        right_layout.addWidget(self.frame_info_label)
        
        # Video display
        self.video_label = InteractiveVideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setMouseTracking(True)
        self.video_label.mouse_pressed.connect(self.on_video_mouse_press)
        self.video_label.mouse_moved.connect(self.on_video_mouse_move)
        self.video_label.mouse_released.connect(self.on_video_mouse_release)
        right_layout.addWidget(self.video_label, 1)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.valueChanged.connect(self.seek_frame)
        right_layout.addWidget(self.progress_slider)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        controls_layout.addWidget(self.stop_btn)
        
        skip_back_btn = QPushButton("???? -10")
        skip_back_btn.clicked.connect(lambda: self.skip_frames(-10))
        controls_layout.addWidget(skip_back_btn)
        
        prev_frame_btn = QPushButton("?? -1")
        prev_frame_btn.clicked.connect(lambda: self.skip_frames(-1))
        controls_layout.addWidget(prev_frame_btn)
        
        next_frame_btn = QPushButton("1 ??")
        next_frame_btn.clicked.connect(lambda: self.skip_frames(1))
        controls_layout.addWidget(next_frame_btn)
        
        skip_forward_btn = QPushButton("+10 ????")
        skip_forward_btn.clicked.connect(lambda: self.skip_frames(10))
        controls_layout.addWidget(skip_forward_btn)
        
        right_layout.addLayout(controls_layout)

        main_layout.addWidget(right_panel, stretch=1)
        main_layout.addWidget(left_scroll)
        
    def browse_video(self):
        """Open file dialog to select video"""
        # Use existing path in text field if available
        start_dir = self.last_video_dir
        current_text = self.video_input.text().strip()
        if current_text:
            current_path = Path(current_text)
            if current_path.exists():
                # If it's a file, use its parent directory
                start_dir = str(current_path.parent if current_path.is_file() else current_path)
            elif current_path.parent.exists():
                # If file doesn't exist but parent does, use parent
                start_dir = str(current_path.parent)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.video_input.setText(file_path)
            self.last_video_dir = str(Path(file_path).parent)
            
            # Try to auto-load DLC predictions
            self.try_autoload_dlc(file_path)
            
    def browse_dlc(self):
        """Open file dialog to select DLC file"""
        # Use existing path in text field if available
        start_dir = self.last_dlc_dir
        current_text = self.dlc_input.text().strip()
        if current_text:
            current_path = Path(current_text)
            if current_path.exists():
                # If it's a file, use its parent directory
                start_dir = str(current_path.parent if current_path.is_file() else current_path)
            elif current_path.parent.exists():
                # If file doesn't exist but parent does, use parent
                start_dir = str(current_path.parent)
        elif self.video_input.text():
            # Fall back to video folder if available, check for predictions folder
            video_path = Path(self.video_input.text())
            video_dir = video_path.parent
            predictions_dir = video_dir / "predictions"
            
            # If predictions folder exists and has DLC files, use it
            if predictions_dir.exists() and predictions_dir.is_dir():
                dlc_files = list(predictions_dir.glob("*.parquet")) + list(predictions_dir.glob("*.csv"))
                if len(dlc_files) > 0:
                    start_dir = str(predictions_dir)
                else:
                    start_dir = str(video_dir)
            else:
                start_dir = str(video_dir)
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DLC File", start_dir,
            "DLC Files (*.parquet *.csv);;All Files (*)"
        )
        if file_path:
            self.dlc_input.setText(file_path)
            self.last_dlc_dir = str(Path(file_path).parent)
            
    def browse_annotation(self):
        """Open file dialog to select annotation file"""
        # Use existing path in text field if available
        start_dir = self.last_video_dir
        current_text = self.annotation_input.text().strip()
        if current_text:
            current_path = Path(current_text)
            if current_path.exists():
                # If it's a file, use its parent directory
                start_dir = str(current_path.parent if current_path.is_file() else current_path)
            elif current_path.parent.exists():
                # If file doesn't exist but parent does, use parent
                start_dir = str(current_path.parent)
        elif self.video_input.text():
            # Fall back to video folder if available
            start_dir = str(Path(self.video_input.text()).parent)
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Annotation File", start_dir,
            "Data Files (*.csv *.parquet *.json);;All Files (*)"
        )
        if file_path:
            self.annotation_input.setText(file_path)
    
    def browse_retrain_config(self):
        """Select DeepLabCut config.yaml for retraining."""
        start_dir = self.last_dlc_dir if Path(self.last_dlc_dir).exists() else str(Path.home())
        current_text = self.retrain_config_input.text().strip()
        if current_text and Path(current_text).exists():
            start_dir = str(Path(current_text).parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DeepLabCut config.yaml",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.retrain_config_input.setText(file_path)
            self.save_preferences()
    
    def browse_run_model_script(self):
        """Select PreyTouch Arena/run_model.py for optional one-click rerun."""
        start_dir = self.last_video_dir if Path(self.last_video_dir).exists() else str(Path.home())
        current_text = self.run_model_script_input.text().strip()
        if current_text and Path(current_text).exists():
            start_dir = str(Path(current_text).parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select run_model.py",
            start_dir,
            "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.run_model_script_input.setText(file_path)
            self.save_preferences()

    def browse_retrain_model_path(self):
        """Select model folder used for optional rerun config update."""
        default_dir = self.default_model_root.expanduser()
        start_dir = str(default_dir if default_dir.exists() else Path.home())

        current_text = self.retrain_model_path_input.text().strip()
        if current_text:
            current_path = Path(current_text).expanduser()
            if current_path.exists():
                start_dir = str(current_path if current_path.is_dir() else current_path.parent)
            elif current_path == default_dir:
                start_dir = str(Path.home())
            elif current_path.parent.exists():
                start_dir = str(current_path.parent)

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select model folder",
            start_dir
        )
        if folder_path:
            self.retrain_model_path_input.setText(folder_path)
            self.save_preferences()

    def browse_calibration_dir(self):
        """Select calibration folder used when recalibrating edited DLC files."""
        start_dir = str(Path.home())
        current_text = self.calibration_dir_input.text().strip()
        if current_text:
            current_path = Path(current_text).expanduser()
            if current_path.exists():
                start_dir = str(current_path if current_path.is_dir() else current_path.parent)
            elif current_path.parent.exists():
                start_dir = str(current_path.parent)

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select calibration folder",
            start_dir
        )
        if folder_path:
            self.calibration_dir_input.setText(folder_path)
            self.calibration_preset_combo.setCurrentIndex(0)
            self.save_preferences()

    def browse_manual_labels_root(self):
        """Select shared root folder used to store manual labels from all edited videos."""
        start_dir = str(Path.home())
        current_text = self.manual_labels_root_input.text().strip()
        if current_text:
            current_path = Path(current_text).expanduser()
            if current_path.exists():
                start_dir = str(current_path if current_path.is_dir() else current_path.parent)
            elif current_path.parent.exists():
                start_dir = str(current_path.parent)

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select shared manual labels root",
            start_dir
        )
        if folder_path:
            self.manual_labels_root_input.setText(folder_path)
            # Refresh on-screen export path for currently loaded video.
            self.init_manual_export_paths(create_dirs=False)
            self.save_preferences()

    def on_manual_labels_root_changed(self):
        """Persist manually typed labels root and refresh current export path preview."""
        self.init_manual_export_paths(create_dirs=False)
        self.save_preferences()
    
    def open_export_folder(self):
        """Open manual_labels export folder in file browser."""
        self.init_manual_export_paths(create_dirs=True)
        if self.export_root is None:
            self.set_edit_status("Export folder not available yet. Load files first.", is_error=True)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_root)))
    
    def open_dlc_folder(self):
        """Open folder containing the active DLC file."""
        dlc_path_text = self.dlc_path or self.dlc_input.text().strip()
        if not dlc_path_text:
            self.set_edit_status("No DLC file selected.", is_error=True)
            return
        dlc_file = Path(dlc_path_text)
        folder = dlc_file.parent if dlc_file.parent.exists() else dlc_file
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
    
    def on_speed_change(self):
        """Handle playback speed changes"""
        speed_value = self.speed_slider.value()
        self.playback_speed = speed_value / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        # Update timer interval if playing
        if self.is_playing and self.fps > 0:
            interval = int(1000 / (self.fps * self.playback_speed))
            self.timer.setInterval(interval)
    
    def set_edit_status(self, message: str, is_error: bool = False):
        """Update lightweight status text for manual edit mode."""
        color = "#A91E2C" if is_error else "#2D5A27"
        self.edit_status_label.setStyleSheet(f"font-size: 9pt; color: {color};")
        self.edit_status_label.setText(message)
    
    def toggle_edit_mode(self):
        """Enable/disable interactive point editing."""
        self.edit_mode = self.edit_mode_btn.isChecked()
        if self.edit_mode:
            if self.cap is None or self.dlc_data is None:
                self.edit_mode = False
                self.edit_mode_btn.setChecked(False)
                self.edit_mode_btn.setText("Edit: OFF")
                self.set_edit_status("Load video + DLC first, then enable Edit mode.", is_error=True)
                return
            self.pause_video()
            self.edit_mode_btn.setText("Edit: ON")
            self.edit_mode_btn.setStyleSheet(
                "QPushButton { background-color: #2E7D32; color: white; font-weight: bold; }"
            )
            self.video_label.setCursor(Qt.CrossCursor)
            self.set_edit_status("Edit mode ON. Drag points, or hold Shift to move all points together. Release queues save and exports frame label.")
        else:
            self.edit_mode_btn.setText("Edit: OFF")
            self.edit_mode_btn.setStyleSheet("")
            self.clear_drag_state()
            self.persist_dlc_edits()
            self.set_edit_status("Edit mode OFF.")
    
    def get_configured_manual_labels_root(self, create_dirs: bool = False) -> Optional[Path]:
        """Resolve shared manual labels root from UI input."""
        root_text = self.manual_labels_root_input.text().strip() if hasattr(self, "manual_labels_root_input") else ""
        root_path = Path(root_text).expanduser() if root_text else self.default_manual_labels_root.expanduser()
        try:
            if root_path.exists():
                if not root_path.is_dir():
                    self.set_edit_status("Manual labels root is not a directory.", is_error=True)
                    return None
                return root_path.resolve()
            if create_dirs:
                root_path.mkdir(parents=True, exist_ok=True)
                return root_path.resolve()
            return root_path
        except Exception as e:
            self.set_edit_status(f"Manual labels root unavailable: {e}", is_error=True)
            return None

    @staticmethod
    def _safe_name_token(text: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)
        safe = safe.strip("_")
        return safe or "video"

    def get_current_video_export_prefix(self) -> str:
        """Build stable per-video prefix to avoid filename collisions in shared export pool."""
        if not self.video_path:
            return "video"
        video_path_obj = Path(self.video_path).expanduser()
        try:
            video_canonical = str(video_path_obj.resolve())
        except Exception:
            video_canonical = str(video_path_obj)
        video_stem = self._safe_name_token(video_path_obj.stem or "video")
        video_token = sha1(video_canonical.encode("utf-8")).hexdigest()[:10]
        return f"{video_stem}__{video_token}"

    @staticmethod
    def find_export_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
        for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            img = images_dir / f"{stem}{suffix}"
            if img.exists():
                return img
        return None

    def flatten_manual_labels_root(self, labels_root: Path, force: bool = False) -> Tuple[int, int]:
        """
        Migrate older per-video manual-label folders into one shared pool:
          labels_root/images/train + labels_root/labels/train
        """
        try:
            labels_root = labels_root.resolve()
        except Exception:
            labels_root = labels_root
        if not force and labels_root in self._flattened_manual_labels_roots:
            return 0, 0

        target_images = labels_root / "images" / "train"
        target_labels = labels_root / "labels" / "train"
        target_log = labels_root / "edits_log.csv"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        if not target_log.exists():
            with open(target_log, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "video_path",
                    "dlc_path",
                    "frame",
                    "dlc_row",
                    "edited_point",
                    "image_path",
                    "label_path",
                ])

        image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        moved_pairs = 0
        moved_logs = 0

        try:
            target_labels_resolved = target_labels.resolve()
        except Exception:
            target_labels_resolved = target_labels
        try:
            labels_root_resolved = labels_root.resolve()
        except Exception:
            labels_root_resolved = labels_root

        source_sets = []
        for labels_dir in sorted(labels_root.rglob("labels/train")):
            try:
                labels_dir_resolved = labels_dir.resolve()
            except Exception:
                labels_dir_resolved = labels_dir
            if labels_dir_resolved == target_labels_resolved:
                continue
            source_root = labels_dir.parent.parent
            try:
                source_root_resolved = source_root.resolve()
            except Exception:
                source_root_resolved = source_root
            if source_root_resolved == labels_root_resolved:
                continue
            images_dir = source_root / "images" / "train"
            if not images_dir.exists():
                continue
            source_sets.append((source_root, images_dir, labels_dir))

        for source_root, source_images, source_labels in source_sets:
            source_tag = self._safe_name_token(source_root.name or "source")

            for label_file in sorted(source_labels.glob("*.txt")):
                image_file = self.find_export_image_for_stem(source_images, label_file.stem)
                if image_file is None:
                    continue

                base_stem = label_file.stem
                candidate_stem = base_stem
                dup_idx = 1
                while True:
                    label_conflict = (target_labels / f"{candidate_stem}.txt").exists()
                    image_conflict = any(
                        (target_images / f"{candidate_stem}{suffix}").exists()
                        for suffix in image_suffixes
                    )
                    if not label_conflict and not image_conflict:
                        break
                    candidate_stem = f"{base_stem}__{source_tag}_{dup_idx:03d}"
                    dup_idx += 1

                target_label_file = target_labels / f"{candidate_stem}.txt"
                target_image_file = target_images / f"{candidate_stem}{image_file.suffix.lower()}"
                shutil.move(str(label_file), str(target_label_file))
                shutil.move(str(image_file), str(target_image_file))
                moved_pairs += 1

            source_log = source_root / "edits_log.csv"
            if source_log.exists():
                with open(target_log, "a", newline="", encoding="utf-8") as dst_f:
                    writer = csv.writer(dst_f)
                    with open(source_log, "r", newline="", encoding="utf-8") as src_f:
                        reader = csv.reader(src_f)
                        _ = next(reader, None)
                        for row in reader:
                            if row:
                                writer.writerow(row)
                source_log.unlink(missing_ok=True)
                moved_logs += 1

            cleanup_dirs = [
                source_labels,
                source_labels.parent,
                source_images,
                source_images.parent,
                source_root,
            ]
            for cleanup_dir in cleanup_dirs:
                try:
                    if cleanup_dir.exists() and cleanup_dir.is_dir():
                        cleanup_dir.rmdir()
                except OSError:
                    pass

        self._flattened_manual_labels_roots.add(labels_root)
        return moved_pairs, moved_logs

    def init_manual_export_paths(self, create_dirs: bool = True):
        """Set output paths for edited frames/labels under shared root; optionally create folders."""
        if not self.video_path:
            return
        labels_root = self.get_configured_manual_labels_root(create_dirs=create_dirs)
        if labels_root is None:
            self.export_root = None
            self.export_frames_dir = None
            self.export_labels_dir = None
            self.export_log_path = None
            return

        self.export_root = labels_root
        self.export_frames_dir = labels_root / "images" / "train"
        self.export_labels_dir = labels_root / "labels" / "train"
        self.export_log_path = labels_root / "edits_log.csv"

        if create_dirs:
            self.export_frames_dir.mkdir(parents=True, exist_ok=True)
            self.export_labels_dir.mkdir(parents=True, exist_ok=True)
            self.export_log_path.parent.mkdir(parents=True, exist_ok=True)

            if not self.export_log_path.exists():
                with open(self.export_log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "video_path",
                        "dlc_path",
                        "frame",
                        "dlc_row",
                        "edited_point",
                        "image_path",
                        "label_path",
                    ])
            moved_pairs, moved_logs = self.flatten_manual_labels_root(labels_root)
            if moved_pairs > 0 or moved_logs > 0:
                self.set_edit_status(
                    f"Merged old per-video exports into shared pool ({moved_pairs} samples, {moved_logs} logs)."
                )
        self.export_path_label.setText(f"Export folder: {self.export_root}")
        dlc_text = self.dlc_path if self.dlc_path else "(load files)"
        self.dlc_save_path_label.setText(f"DLC file overwritten on save: {dlc_text}")

    def get_manual_labels_pool_root(self) -> Optional[Path]:
        """Return shared manual labels root across all edited videos."""
        return self.get_configured_manual_labels_root(create_dirs=False)

    def begin_edit_revision(self) -> int:
        """Advance revision counter for a new in-memory edit operation."""
        self._edit_revision += 1
        return int(self._edit_revision)

    def queue_pending_export(self, frame_num: int, tags: Any, revision: int):
        """Mark a frame for later image/label export and remember its latest edit revision."""
        frame_num = int(frame_num)
        frame_exports = self.pending_export_frames.setdefault(frame_num, set())
        if isinstance(tags, str):
            frame_exports.add(tags)
        else:
            frame_exports.update(tags)
        self.pending_export_frame_revisions[frame_num] = max(
            int(revision),
            int(self.pending_export_frame_revisions.get(frame_num, -1)),
        )

    def set_save_ui_busy(self, busy: bool, export_frames: bool = False, autosave: bool = False):
        """Reflect background save state in the Save button without blocking editing."""
        if not hasattr(self, "save_edits_btn"):
            return
        self.save_edits_btn.setEnabled(not busy)
        if not busy:
            self.save_edits_btn.setText("Save Edits Now")
            return
        if export_frames:
            self.save_edits_btn.setText("Saving + Exporting...")
        elif autosave:
            self.save_edits_btn.setText("Autosaving...")
        else:
            self.save_edits_btn.setText("Saving...")

    def build_save_job(self, export_frames: bool, autosave: bool) -> Dict[str, Any]:
        """Snapshot the current state for background DLC save/export work."""
        if export_frames:
            self.init_manual_export_paths(create_dirs=False)

        dlc_file = Path(self.dlc_path)
        backup_file = dlc_file.with_suffix(dlc_file.suffix + ".bak")
        return {
            "revision": int(self._edit_revision),
            "dlc_path": self.dlc_path,
            "is_parquet": self.dlc_path.lower().endswith(".parquet"),
            "dlc_data": self.dlc_data.copy(deep=True),
            "create_backup": (not self._dlc_backup_created) and dlc_file.exists() and not backup_file.exists(),
            "export_frames": bool(export_frames),
            "autosave": bool(autosave),
            "pending_export_frames": (
                {int(frame_num): sorted(tags) for frame_num, tags in self.pending_export_frames.items()}
                if export_frames else {}
            ),
            "current_frame": int(self.current_frame),
            "current_frame_bgr": (
                self._last_rendered_bgr.copy()
                if export_frames and self._last_rendered_bgr is not None else None
            ),
            "video_path": self.video_path,
            "export_root": str(self.export_root) if self.export_root is not None else None,
            "export_frames_dir": str(self.export_frames_dir) if self.export_frames_dir is not None else None,
            "export_labels_dir": str(self.export_labels_dir) if self.export_labels_dir is not None else None,
            "export_log_path": str(self.export_log_path) if self.export_log_path is not None else None,
            "video_export_prefix": self.get_current_video_export_prefix(),
            "editable_point_names": sorted(self.editable_point_names),
            "coords_are_normalized": bool(self.coords_are_normalized),
            "dlc_frame_map": dict(self.dlc_frame_map) if self.dlc_frame_map is not None else None,
            "start_frame": int(self.start_frame),
        }

    def launch_save_worker(self, job: Dict[str, Any]):
        """Start a background worker that writes DLC edits and optional exports."""
        self._save_in_progress = True
        self.set_save_ui_busy(
            True,
            export_frames=bool(job["export_frames"]),
            autosave=bool(job["autosave"]),
        )

        self._save_thread = QThread(self)
        self._save_worker = SaveExportWorker(job)
        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.finished.connect(self.on_save_worker_finished)
        self._save_worker.failed.connect(self.on_save_worker_failed)
        self._save_worker.finished.connect(self._save_thread.quit)
        self._save_worker.failed.connect(self._save_thread.quit)
        self._save_worker.finished.connect(self._save_worker.deleteLater)
        self._save_worker.failed.connect(self._save_worker.deleteLater)
        self._save_thread.finished.connect(self._save_thread.deleteLater)
        self._save_thread.finished.connect(self.on_save_thread_finished)
        self._save_thread.start()

    def on_save_thread_finished(self):
        """Drop worker references after the background thread shuts down."""
        self._save_thread = None
        self._save_worker = None

    @staticmethod
    def _path_signature(path_text: Optional[str]) -> Optional[Tuple[int, int]]:
        """Return a lightweight on-disk signature for change detection."""
        if not path_text:
            return None
        try:
            stat = Path(path_text).expanduser().stat()
        except Exception:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))

    @staticmethod
    def _same_path(path_a: Optional[str], path_b: Optional[str]) -> bool:
        """Best-effort path equality check across user-entered paths."""
        if not path_a or not path_b:
            return False
        try:
            return Path(path_a).expanduser().resolve() == Path(path_b).expanduser().resolve()
        except Exception:
            return str(Path(path_a).expanduser()) == str(Path(path_b).expanduser())

    def remember_loaded_dlc_signature(self, path_text: Optional[str] = None):
        """Remember the on-disk DLC state after a successful load/save."""
        self._loaded_dlc_signature = self._path_signature(path_text or self.dlc_path)

    def dlc_changed_on_disk(self, path_text: Optional[str] = None) -> bool:
        """Detect external DLC modifications since the last load/save."""
        target_path = path_text or self.dlc_path
        if not target_path or self._loaded_dlc_signature is None:
            return False
        return self._path_signature(target_path) != self._loaded_dlc_signature

    def clear_pending_save_state(self):
        """Drop queued save/export state without mutating loaded DLC data."""
        if self.save_timer.isActive():
            self.save_timer.stop()
        self.pending_export_frames = {}
        self.pending_export_frame_revisions = {}
        self._pending_save = False
        self._save_followup_requested = False
        self._save_followup_export = False

    def note_unsaved_edit(self, message: str, rerender_preview: bool = False):
        """Mark edits as pending and make it explicit that nothing was autosaved."""
        if self.save_timer.isActive():
            self.save_timer.stop()
        if rerender_preview:
            self.render_drag_preview()
        self.set_edit_status(f"{message} Not saved yet. Click Save Edits Now to write DLC + exports.")

    def confirm_discard_unsaved_reload(self, message: str) -> bool:
        """Ask before discarding stale in-memory edits to reload from disk."""
        reply = QMessageBox.warning(
            self,
            "Reload From Disk",
            message,
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Yes

    def queue_load_after_save(self, video_path: str, dlc_path: str, message: str):
        """Save current work first, then load the requested files."""
        self._load_after_save = True
        self._load_after_save_paths = (video_path, dlc_path)
        self.set_edit_status(message, is_error=False)
        if not self._save_in_progress:
            self.persist_dlc_edits(export_frames=True, autosave=False)

    def maybe_continue_after_save(self):
        """Run queued follow-up work after a save/export worker finishes."""
        if self._save_in_progress:
            return

        if self._save_followup_requested:
            export_frames = bool(self._save_followup_export)
            self._save_followup_requested = False
            self._save_followup_export = False
            QTimer.singleShot(
                0,
                lambda: self.persist_dlc_edits(
                    export_frames=export_frames,
                    autosave=not export_frames,
                ),
            )
            return

        if self._load_after_save:
            if self._pending_save or self.pending_export_frames:
                QTimer.singleShot(0, lambda: self.persist_dlc_edits(export_frames=True, autosave=False))
            else:
                self._load_after_save = False
                pending_paths = self._load_after_save_paths
                self._load_after_save_paths = None
                if pending_paths is not None:
                    video_path, dlc_path = pending_paths
                    self.video_input.setText(video_path)
                    self.dlc_input.setText(dlc_path)
                    QTimer.singleShot(0, self.load_files)
            return

        if self._close_after_save:
            if self._pending_save or self.pending_export_frames:
                QTimer.singleShot(0, lambda: self.persist_dlc_edits(export_frames=True, autosave=False))
            else:
                self.close()
            return

        if self._calibration_after_save:
            if self._pending_save or self.pending_export_frames:
                QTimer.singleShot(0, lambda: self.persist_dlc_edits(export_frames=True, autosave=False))
            else:
                self._calibration_after_save = False
                QTimer.singleShot(0, self.start_reapply_calibration)
            return

        if self._retrain_after_save:
            if self._pending_save or self.pending_export_frames:
                QTimer.singleShot(0, lambda: self.persist_dlc_edits(export_frames=True, autosave=False))
            else:
                self._retrain_after_save = False
                QTimer.singleShot(0, self.start_retrain_and_rerun)

    def on_save_worker_finished(self, result: Dict[str, Any]):
        """Apply background save/export results back onto UI state."""
        self._save_in_progress = False
        self.set_save_ui_busy(False)

        if result.get("created_backup"):
            self._dlc_backup_created = True

        result_dlc_path = result.get("dlc_path")
        snapshot_revision = int(result.get("revision", -1))
        export_frames = bool(result.get("export_frames"))
        autosave = bool(result.get("autosave"))
        processed_frames = [int(frame_num) for frame_num in result.get("processed_frames", [])]

        if result_dlc_path == self.dlc_path:
            if export_frames:
                for frame_num in processed_frames:
                    frame_revision = int(self.pending_export_frame_revisions.get(frame_num, -1))
                    if frame_revision <= snapshot_revision:
                        self.pending_export_frames.pop(frame_num, None)
                        self.pending_export_frame_revisions.pop(frame_num, None)
            if self._edit_revision <= snapshot_revision:
                self._pending_save = False
            self.remember_loaded_dlc_signature(result_dlc_path)

        dlc_file = result_dlc_path or self.dlc_path
        if export_frames:
            export_root_text = result.get("export_root") or "(not set)"
            self.set_edit_status(
                f"Saved edits to {dlc_file} ({int(result.get('exported_count', 0))} frame exports) -> {export_root_text}"
            )
        elif autosave:
            self.set_edit_status(
                f"Autosaved edits to {dlc_file} (data only; frame exports wait for Save Edits Now)."
            )
        else:
            self.set_edit_status(f"Saved edits to {dlc_file}.")

        self.maybe_continue_after_save()

    def on_save_worker_failed(self, error_text: str):
        """Surface background save/export failures without crashing the UI."""
        self._save_in_progress = False
        self.set_save_ui_busy(False)
        self._save_followup_requested = False
        self._save_followup_export = False
        self._load_after_save = False
        self._load_after_save_paths = None
        self._close_after_save = False
        self._retrain_after_save = False
        short_error = error_text.strip().splitlines()[-1] if error_text.strip() else "unknown error"
        self.set_edit_status(f"Failed saving DLC file: {short_error}", is_error=True)
        print("Error saving edited DLC file:")
        print(error_text)
    
    def label_to_frame_coords(self, label_x: int, label_y: int) -> Optional[Tuple[int, int]]:
        """Map click coordinates from QLabel space into original frame pixel space."""
        if self._display_rect is None or self._last_rendered_frame_size is None:
            return None
        
        x0, y0, disp_w, disp_h = self._display_rect
        if disp_w <= 0 or disp_h <= 0:
            return None
        if label_x < x0 or label_y < y0 or label_x >= x0 + disp_w or label_y >= y0 + disp_h:
            return None
        
        frame_w, frame_h = self._last_rendered_frame_size
        rel_x = (label_x - x0) / disp_w
        rel_y = (label_y - y0) / disp_h
        
        frame_x = int(round(rel_x * max(frame_w - 1, 1)))
        frame_y = int(round(rel_y * max(frame_h - 1, 1)))
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        return frame_x, frame_y
    
    def frame_to_label_coords(self, frame_x: int, frame_y: int) -> Optional[Tuple[int, int]]:
        """Map frame pixel coordinates back into QLabel coordinate space."""
        if self._display_rect is None or self._last_rendered_frame_size is None:
            return None
        x0, y0, disp_w, disp_h = self._display_rect
        frame_w, frame_h = self._last_rendered_frame_size
        if frame_w <= 0 or frame_h <= 0 or disp_w <= 0 or disp_h <= 0:
            return None
        rel_x = float(frame_x) / float(max(frame_w - 1, 1))
        rel_y = float(frame_y) / float(max(frame_h - 1, 1))
        label_x = int(round(x0 + rel_x * disp_w))
        label_y = int(round(y0 + rel_y * disp_h))
        return label_x, label_y
    
    def point_raw_to_pixels(
        self, x_raw: float, y_raw: float, frame_w: int, frame_h: int
    ) -> Tuple[int, int]:
        """Convert stored DLC coordinates to frame pixel coordinates."""
        if self.coords_are_normalized:
            x_px = int(float(x_raw) * frame_w)
            y_px = int(float(y_raw) * frame_h)
        else:
            x_px = int(float(x_raw))
            y_px = int(float(y_raw))
        x_px = max(0, min(frame_w - 1, x_px))
        y_px = max(0, min(frame_h - 1, y_px))
        return x_px, y_px
    
    def point_pixels_to_raw(
        self, x_px: int, y_px: int, frame_w: int, frame_h: int
    ) -> Tuple[float, float]:
        """Convert frame pixel coordinates back into DLC storage format."""
        if self.coords_are_normalized:
            x_raw = float(x_px) / float(max(frame_w, 1))
            y_raw = float(y_px) / float(max(frame_h, 1))
            x_raw = min(max(x_raw, 0.0), 1.0)
            y_raw = min(max(y_raw, 0.0), 1.0)
        else:
            x_raw = float(x_px)
            y_raw = float(y_px)
        return x_raw, y_raw
    
    def _column_index(self, col_name: str) -> Optional[int]:
        """Resolve a column name to a stable integer index."""
        if self.dlc_data is None:
            return None
        if col_name not in self.dlc_data.columns:
            return None
        loc = self.dlc_data.columns.get_loc(col_name)
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, np.ndarray):
            if loc.dtype == bool:
                idx = np.flatnonzero(loc)
                return int(idx[0]) if len(idx) > 0 else None
            return int(loc[0]) if len(loc) > 0 else None
        if isinstance(loc, list):
            return int(loc[0]) if len(loc) > 0 else None
        return int(loc)
    
    def set_cell_value(self, row_idx: int, col_name: str, value) -> bool:
        """Set one dataframe cell by row index and column name."""
        col_idx = self._column_index(col_name)
        if col_idx is None:
            return False
        self.dlc_data.iat[int(row_idx), col_idx] = value
        return True
    
    def get_cell_value(self, row_idx: int, col_name: str):
        """Read one dataframe cell by row index and column name."""
        col_idx = self._column_index(col_name)
        if col_idx is None:
            return np.nan
        return self.dlc_data.iat[int(row_idx), col_idx]

    def get_point_confidence_columns(self, point_name: str) -> List[str]:
        """Return confidence column names for a point (supports *_cam fallback)."""
        if self.dlc_data is None:
            return []
        candidates = [point_name]
        if point_name.endswith("_cam"):
            candidates.append(point_name[:-4])
        cols = []
        seen = set()
        for candidate in candidates:
            for suffix in ("_prob", "_likelihood", "_conf"):
                conf_col = f"{candidate}{suffix}"
                if conf_col in self.dlc_data.columns and conf_col not in seen:
                    cols.append(conf_col)
                    seen.add(conf_col)
        return cols

    def get_point_confidence_value_from_data(self, row_data, point_name: str) -> Optional[float]:
        """Read the first non-NaN confidence value available for a point from cached row data."""
        for conf_col in self.get_point_confidence_columns(point_name):
            try:
                conf_val = row_data.get(conf_col, np.nan)
            except Exception:
                conf_val = np.nan
            if pd.notna(conf_val):
                try:
                    return float(conf_val)
                except Exception:
                    continue
        return None

    def get_point_confidence_value(self, row_idx: int, point_name: str) -> Optional[float]:
        """Read the first non-NaN confidence value available for a point."""
        if self.dlc_data is None:
            return None
        return self.get_point_confidence_value_from_data(self.dlc_data.iloc[int(row_idx)], point_name)

    def ensure_confidence_columns(self, point_names: Optional[List[str]] = None) -> int:
        """Create per-point *_conf columns only when a point has no confidence storage at all."""
        if self.dlc_data is None:
            return 0
        if point_names is None:
            point_names = sorted(self.editable_point_names)

        created = 0
        for point_name in point_names:
            if self.get_point_confidence_columns(point_name):
                continue
            conf_col = f"{point_name}_conf"
            if conf_col in self.dlc_data.columns:
                continue
            self.dlc_data[conf_col] = np.nan
            created += 1
        return created

    def set_points_confidence_for_rows(
        self, row_indices: List[int], point_names, confidence_value: float
    ) -> int:
        """Set confidence columns for given point names across one or more rows."""
        if self.dlc_data is None:
            return 0
        rows = [int(r) for r in row_indices]
        if not rows:
            return 0

        col_indices = []
        seen_cols = set()
        for point_name in point_names:
            for conf_col in self.get_point_confidence_columns(point_name):
                conf_idx = self._column_index(conf_col)
                if conf_idx is None or conf_idx in seen_cols:
                    continue
                col_indices.append(conf_idx)
                seen_cols.add(conf_idx)
        if not col_indices:
            return 0

        conf_val = float(confidence_value)
        if len(rows) == 1:
            self.dlc_data.iloc[rows[0], col_indices] = conf_val
        else:
            self.dlc_data.iloc[rows, col_indices] = np.full((len(rows), len(col_indices)), conf_val)
        return len(col_indices)

    def find_ear_point_pair(self) -> Optional[Tuple[str, str]]:
        """Detect a left/right ear pair among editable points."""
        if not self.editable_point_names:
            return None

        original_by_lower = {point_name.lower(): point_name for point_name in self.editable_point_names}
        for point_name in sorted(self.editable_point_names):
            point_name_lower = point_name.lower()
            if "ear" not in point_name_lower or "left" not in point_name_lower:
                continue

            candidate_lowers = []
            if "left_" in point_name_lower:
                candidate_lowers.append(point_name_lower.replace("left_", "right_", 1))
            if "_left" in point_name_lower:
                candidate_lowers.append(point_name_lower.replace("_left", "_right", 1))
            candidate_lowers.append(point_name_lower.replace("left", "right", 1))

            for candidate_lower in candidate_lowers:
                candidate = original_by_lower.get(candidate_lower)
                if candidate is not None and "ear" in candidate_lower:
                    return point_name, candidate
        return None

    def refresh_flip_ears_button(self):
        """Enable ear-swap only when the current DLC file has a left/right ear pair."""
        if not hasattr(self, "flip_ears_btn"):
            return
        ear_pair = self.find_ear_point_pair()
        self.flip_ears_btn.setEnabled(ear_pair is not None)
        if hasattr(self, "flip_ears_range_btn"):
            self.flip_ears_range_btn.setEnabled(ear_pair is not None)
        if ear_pair is None:
            self.flip_ears_btn.setToolTip("No left/right ear pair detected in the current DLC file.")
            if hasattr(self, "flip_ears_range_btn"):
                self.flip_ears_range_btn.setToolTip("No left/right ear pair detected in the current DLC file.")
        else:
            self.flip_ears_btn.setToolTip(f"Swap {ear_pair[0]} and {ear_pair[1]} on the current frame.")
            if hasattr(self, "flip_ears_range_btn"):
                self.flip_ears_range_btn.setToolTip(
                    f"Swap {ear_pair[0]} and {ear_pair[1]} across an inclusive frame range."
                )
    
    def find_nearest_point_for_edit(self, dlc_row_idx: int, frame_x: int, frame_y: int) -> Optional[str]:
        """Find nearest visible point to cursor for drag selection."""
        if self._last_rendered_frame_size is None or self.dlc_data is None:
            return None
        
        frame_w, frame_h = self._last_rendered_frame_size
        row = self.dlc_data.iloc[dlc_row_idx]
        best_name = None
        effective_radius = self.drag_radius_px
        if self._display_rect is not None:
            _, _, disp_w, disp_h = self._display_rect
            if disp_w > 0 and disp_h > 0:
                # Keep grab radius stable in screen pixels for downscaled videos.
                scale_x = frame_w / disp_w
                scale_y = frame_h / disp_h
                effective_radius = max(self.drag_radius_px, int(round(16.0 * max(scale_x, scale_y))))
        best_dist_sq = effective_radius * effective_radius
        
        for point_name, config in self.point_configs.items():
            if point_name not in self.editable_point_names:
                continue
            if not config.get("enabled", True):
                continue
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                continue
            x_val = row[x_col]
            y_val = row[y_col]
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            
            x_px, y_px = self.point_raw_to_pixels(float(x_val), float(y_val), frame_w, frame_h)
            dx = frame_x - x_px
            dy = frame_y - y_px
            dist_sq = dx * dx + dy * dy
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_name = point_name
        
        return best_name

    def get_editable_points_raw_for_row(self, dlc_row_idx: int) -> Dict[str, Tuple[float, float]]:
        """Return editable points with valid coordinates for one DLC row."""
        if self.dlc_data is None:
            return {}

        row = self.dlc_data.iloc[int(dlc_row_idx)]
        points_raw = {}
        for point_name in sorted(self.editable_point_names):
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                continue
            x_val = row[x_col]
            y_val = row[y_col]
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            try:
                points_raw[point_name] = (float(x_val), float(y_val))
            except Exception:
                continue
        return points_raw

    def find_nose_point_for_rotation(self) -> Optional[str]:
        """Find the editable nose point used as the rotation center."""
        if not self.editable_point_names:
            return None

        by_lower = {point_name.lower(): point_name for point_name in self.editable_point_names}
        for preferred in ("nose_cam", "nose"):
            if preferred in by_lower:
                return by_lower[preferred]

        for point_name in sorted(self.editable_point_names):
            if "nose" in point_name.lower():
                return point_name
        return None

    def build_rotated_points_preview(self, frame_x: int, frame_y: int) -> Tuple[Dict[str, Tuple[float, float]], float]:
        """Return rotated editable points around the nose center for the current mouse location."""
        if self._last_rendered_frame_size is None:
            return {}, 0.0
        if (
            self.drag_rotation_center_px is None
            or self.drag_rotation_start_angle is None
            or not self.drag_source_points_raw
        ):
            return {}, 0.0

        frame_w, frame_h = self._last_rendered_frame_size
        center_x, center_y = self.drag_rotation_center_px
        lead_dx = float(frame_x) - float(center_x)
        lead_dy = float(frame_y) - float(center_y)
        if math.hypot(lead_dx, lead_dy) < 2.0:
            return {}, 0.0

        current_angle = math.atan2(lead_dy, lead_dx)
        angle_delta = current_angle - float(self.drag_rotation_start_angle)
        cos_a = math.cos(angle_delta)
        sin_a = math.sin(angle_delta)
        preview_points = {}

        for point_name, (x_raw_start, y_raw_start) in self.drag_source_points_raw.items():
            x_px_start, y_px_start = self.point_raw_to_pixels(x_raw_start, y_raw_start, frame_w, frame_h)
            dx = float(x_px_start) - float(center_x)
            dy = float(y_px_start) - float(center_y)
            new_x_px = int(round(float(center_x) + cos_a * dx - sin_a * dy))
            new_y_px = int(round(float(center_y) + sin_a * dx + cos_a * dy))
            new_x_px = max(0, min(frame_w - 1, new_x_px))
            new_y_px = max(0, min(frame_h - 1, new_y_px))
            preview_points[point_name] = self.point_pixels_to_raw(new_x_px, new_y_px, frame_w, frame_h)

        return preview_points, math.degrees(angle_delta)
    
    def update_dragged_point(self, frame_x: int, frame_y: int, commit: bool):
        """Apply point drag updates to in-memory DLC data, and optionally persist/export."""
        if self.dragging_point_name is None or self.dragging_dlc_row_idx is None:
            return
        if self._last_rendered_frame_size is None:
            return
        
        dlc_row_idx = self.dragging_dlc_row_idx
        frame_w, frame_h = self._last_rendered_frame_size
        
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))

        if self.drag_rotate_points:
            preview_points, angle_degrees = self.build_rotated_points_preview(frame_x, frame_y)
            if not preview_points:
                return

            if not commit:
                self.drag_preview_points_raw = preview_points
                self.drag_preview_raw = preview_points.get(self.dragging_point_name)
                return

            if self.drag_preview_points_raw:
                preview_points = dict(self.drag_preview_points_raw)

            rotated_point_names = []
            for point_name, (x_raw, y_raw) in preview_points.items():
                x_col = f"{point_name}_x"
                y_col = f"{point_name}_y"
                if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                    continue
                self.set_cell_value(dlc_row_idx, x_col, x_raw)
                self.set_cell_value(dlc_row_idx, y_col, y_raw)
                rotated_point_names.append(point_name)

            if not rotated_point_names:
                self.set_edit_status("Control-drag found no editable points to rotate.", is_error=True)
                return

            self.set_points_confidence_for_rows(
                [int(dlc_row_idx)], rotated_point_names, self.manual_edit_confidence
            )
            self.drag_preview_raw = None
            self.drag_preview_points_raw = {}

            self._last_dlc_row_idx = None
            edit_revision = self.begin_edit_revision()
            self._pending_save = True
            self.queue_pending_export(int(self.current_frame), rotated_point_names, edit_revision)
            center_label = self.drag_rotation_center_point or "nose"
            self.note_unsaved_edit(
                f"Updated frame {self.current_frame}, rotated {len(rotated_point_names)} point(s) "
                f"around {center_label} by {angle_degrees:.1f} deg "
                f"(conf={self.manual_edit_confidence:.2f})."
            )
            return

        if self.drag_all_points:
            if self.drag_start_frame_pos is None or not self.drag_source_points_raw:
                return

            delta_x = frame_x - int(self.drag_start_frame_pos[0])
            delta_y = frame_y - int(self.drag_start_frame_pos[1])
            preview_points = {}

            for point_name, (x_raw_start, y_raw_start) in self.drag_source_points_raw.items():
                x_px_start, y_px_start = self.point_raw_to_pixels(x_raw_start, y_raw_start, frame_w, frame_h)
                new_x_px = max(0, min(frame_w - 1, x_px_start + delta_x))
                new_y_px = max(0, min(frame_h - 1, y_px_start + delta_y))
                preview_points[point_name] = self.point_pixels_to_raw(new_x_px, new_y_px, frame_w, frame_h)

            if not commit:
                self.drag_preview_points_raw = preview_points
                self.drag_preview_raw = preview_points.get(self.dragging_point_name)
                return

            if self.drag_preview_points_raw:
                preview_points = dict(self.drag_preview_points_raw)

            moved_point_names = []
            for point_name, (x_raw, y_raw) in preview_points.items():
                x_col = f"{point_name}_x"
                y_col = f"{point_name}_y"
                if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                    continue
                self.set_cell_value(dlc_row_idx, x_col, x_raw)
                self.set_cell_value(dlc_row_idx, y_col, y_raw)
                moved_point_names.append(point_name)

            if not moved_point_names:
                self.set_edit_status("Shift-drag found no editable points to update.", is_error=True)
                return

            self.set_points_confidence_for_rows(
                [int(dlc_row_idx)], moved_point_names, self.manual_edit_confidence
            )
            self.drag_preview_raw = None
            self.drag_preview_points_raw = {}

            self._last_dlc_row_idx = None
            edit_revision = self.begin_edit_revision()
            self._pending_save = True
            self.queue_pending_export(int(self.current_frame), moved_point_names, edit_revision)
            self.note_unsaved_edit(
                f"Updated frame {self.current_frame}, moved {len(moved_point_names)} point(s) together "
                f"(dx={delta_x}, dy={delta_y}; conf={self.manual_edit_confidence:.2f})."
            )
            return

        point_name = self.dragging_point_name
        if point_name not in self.editable_point_names:
            return
        
        x_col = f"{point_name}_x"
        y_col = f"{point_name}_y"
        if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
            self.set_edit_status(f"Missing columns for point '{point_name}'.", is_error=True)
            return
        
        x_raw, y_raw = self.point_pixels_to_raw(frame_x, frame_y, frame_w, frame_h)
        if not commit:
            # Keep drag ultra-lightweight: preview coordinates only, no dataframe writes here.
            self.drag_preview_raw = (x_raw, y_raw)
            return
        
        # Commit once on mouse release.
        if self.drag_preview_raw is not None:
            x_raw, y_raw = self.drag_preview_raw
        
        self.set_cell_value(dlc_row_idx, x_col, x_raw)
        self.set_cell_value(dlc_row_idx, y_col, y_raw)
        self.set_points_confidence_for_rows(
            [int(dlc_row_idx)], [point_name], self.manual_edit_confidence
        )
        self.drag_preview_raw = None
        
        # Reset cached row so display uses latest values immediately.
        self._last_dlc_row_idx = None
        
        edit_revision = self.begin_edit_revision()
        self._pending_save = True
        self.queue_pending_export(int(self.current_frame), point_name, edit_revision)
        self.note_unsaved_edit(
            f"Updated frame {self.current_frame}, point '{point_name}' (conf={self.manual_edit_confidence:.2f})."
        )
    
    def on_video_mouse_press(self, event):
        """Start dragging a point in edit mode."""
        if not self.edit_mode:
            return
        if event.button() != Qt.LeftButton:
            return
        if self.dlc_data is None or self.cap is None:
            return
        
        coords = self.label_to_frame_coords(event.pos().x(), event.pos().y())
        if coords is None:
            return
        
        dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
        if dlc_row_idx is None:
            self.set_edit_status("No DLC row for this frame.", is_error=True)
            return
        if not self.editable_point_names:
            self.set_edit_status("No editable points found for this file.", is_error=True)
            return
        
        self.pause_video()
        frame_x, frame_y = coords
        point_name = self.find_nearest_point_for_edit(dlc_row_idx, frame_x, frame_y)
        if point_name is None:
            self.set_edit_status("No nearby point found. Click closer to a dot.", is_error=True)
            return

        ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
        shift_pressed = bool(event.modifiers() & Qt.ShiftModifier) and not ctrl_pressed
        self.dragging_point_name = point_name
        self.dragging_dlc_row_idx = dlc_row_idx
        self.drag_start_frame = self.current_frame
        self.drag_start_frame_pos = (int(frame_x), int(frame_y))
        self.drag_all_points = shift_pressed
        self.drag_rotate_points = ctrl_pressed
        self.drag_source_points_raw = {}
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        self.drag_rotation_center_point = None
        self.drag_rotation_center_px = None
        self.drag_rotation_start_angle = None

        if ctrl_pressed:
            if self._last_rendered_frame_size is None:
                self.clear_drag_state()
                self.set_edit_status("Cannot rotate points before a frame is rendered.", is_error=True)
                return

            self.drag_source_points_raw = self.get_editable_points_raw_for_row(dlc_row_idx)
            if not self.drag_source_points_raw:
                self.clear_drag_state()
                self.set_edit_status("No editable points with coordinates found for Control-drag.", is_error=True)
                return

            nose_point = self.find_nose_point_for_rotation()
            if nose_point is None or nose_point not in self.drag_source_points_raw:
                self.clear_drag_state()
                self.set_edit_status("Control-drag needs a valid nose point to use as the rotation center.", is_error=True)
                return

            frame_w, frame_h = self._last_rendered_frame_size
            center_raw = self.drag_source_points_raw[nose_point]
            center_px = self.point_raw_to_pixels(center_raw[0], center_raw[1], frame_w, frame_h)
            lead_raw = self.drag_source_points_raw.get(point_name)
            if lead_raw is None:
                self.clear_drag_state()
                self.set_edit_status("Control-drag needs a point with valid coordinates to lead the rotation.", is_error=True)
                return

            lead_px = self.point_raw_to_pixels(lead_raw[0], lead_raw[1], frame_w, frame_h)
            lead_dx = float(lead_px[0]) - float(center_px[0])
            lead_dy = float(lead_px[1]) - float(center_px[1])
            if math.hypot(lead_dx, lead_dy) < 2.0:
                self.clear_drag_state()
                self.set_edit_status("Control-drag needs a non-nose point away from the nose.", is_error=True)
                return

            self.drag_rotation_center_point = nose_point
            self.drag_rotation_center_px = center_px
            self.drag_rotation_start_angle = math.atan2(lead_dy, lead_dx)
        elif shift_pressed:
            self.drag_source_points_raw = self.get_editable_points_raw_for_row(dlc_row_idx)
            if not self.drag_source_points_raw:
                self.clear_drag_state()
                self.set_edit_status("No editable points with coordinates found for Shift-drag.", is_error=True)
                return

        self.video_label.setCursor(Qt.ClosedHandCursor)
        color = self.point_configs.get(point_name, {}).get("color", (255, 220, 0))
        self.video_label.set_drag_overlay((event.pos().x(), event.pos().y()), color=color)
        if ctrl_pressed:
            self.set_edit_status(
                f"Control-dragging {len(self.drag_source_points_raw)} point(s) around "
                f"{self.drag_rotation_center_point} on frame {self.current_frame}..."
            )
        elif shift_pressed:
            self.set_edit_status(
                f"Shift-dragging {len(self.drag_source_points_raw)} point(s) together on frame {self.current_frame}..."
            )
        else:
            self.set_edit_status(f"Dragging '{point_name}' on frame {self.current_frame}...")
    
    def on_video_mouse_move(self, event):
        """Update dragged point while mouse moves."""
        if not self.edit_mode:
            return
        if self.dragging_point_name is None:
            return
        if self.drag_start_frame != self.current_frame:
            # Guard against frame changes while dragging.
            return
        
        coords = self.label_to_frame_coords(event.pos().x(), event.pos().y())
        if coords is None:
            return
        
        frame_x, frame_y = coords
        self.update_dragged_point(frame_x, frame_y, commit=False)
        self.render_drag_preview()
        color = self.point_configs.get(self.dragging_point_name, {}).get("color", (255, 220, 0))
        label_coords = self.frame_to_label_coords(frame_x, frame_y)
        if label_coords is not None:
            self.video_label.set_drag_overlay(label_coords, color=color)
    
    def on_video_mouse_release(self, event):
        """Commit final point location on drag release."""
        if event.button() != Qt.LeftButton:
            return
        if self.dragging_point_name is None:
            return
        
        coords = self.label_to_frame_coords(event.pos().x(), event.pos().y())
        if coords is not None:
            frame_x, frame_y = coords
            self.update_dragged_point(frame_x, frame_y, commit=True)
        
        self.clear_drag_state()
        if coords is not None:
            self.display_frame()
    
    def clear_drag_state(self):
        """Reset drag interaction state when frame/navigation changes."""
        self.dragging_point_name = None
        self.dragging_dlc_row_idx = None
        self.drag_start_frame = None
        self.drag_start_frame_pos = None
        self.drag_all_points = False
        self.drag_rotate_points = False
        self.drag_source_points_raw = {}
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        self.drag_rotation_center_point = None
        self.drag_rotation_center_px = None
        self.drag_rotation_start_angle = None
        self.video_label.set_drag_overlay(None)
        self.video_label.setCursor(Qt.CrossCursor if self.edit_mode else Qt.ArrowCursor)
    
    def clear_current_frame_points(self):
        """Mark current frame as no-head by clearing all point coordinates."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return
        
        dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
        if dlc_row_idx is None:
            self.set_edit_status("No DLC row for this frame.", is_error=True)
            return
        
        cleared = 0
        for point_name in self.editable_point_names:
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col in self.dlc_data.columns:
                self.set_cell_value(dlc_row_idx, x_col, np.nan)
            if y_col in self.dlc_data.columns:
                self.set_cell_value(dlc_row_idx, y_col, np.nan)
            if x_col in self.dlc_data.columns or y_col in self.dlc_data.columns:
                cleared += 1
            
            # Best effort confidence cleanup for no-head frames.
            self.set_points_confidence_for_rows([int(dlc_row_idx)], [point_name], 0.0)
        
        self._last_dlc_row_idx = None
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        edit_revision = self.begin_edit_revision()
        self._pending_save = True
        self.queue_pending_export(int(self.current_frame), "NO_HEAD", edit_revision)
        self.note_unsaved_edit(
            f"Frame {self.current_frame}: set to NO_HEAD (cleared {cleared} points).",
            rerender_preview=True,
        )

    def clear_selected_point_current_frame(self):
        """Clear one selected point on the current frame by setting x/y to NaN."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return
        if not self.editable_point_names:
            self.set_edit_status("No editable points found for this file.", is_error=True)
            return

        point_name = self.clear_point_combo.currentText().strip()
        if not point_name or point_name not in self.editable_point_names:
            self.set_edit_status("Select a point to clear.", is_error=True)
            return

        dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
        if dlc_row_idx is None:
            self.set_edit_status("No DLC row for this frame.", is_error=True)
            return

        x_col = f"{point_name}_x"
        y_col = f"{point_name}_y"
        changed = False
        if x_col in self.dlc_data.columns:
            self.set_cell_value(dlc_row_idx, x_col, np.nan)
            changed = True
        if y_col in self.dlc_data.columns:
            self.set_cell_value(dlc_row_idx, y_col, np.nan)
            changed = True
        self.set_points_confidence_for_rows([int(dlc_row_idx)], [point_name], 0.0)

        if not changed:
            self.set_edit_status(f"Missing x/y columns for point '{point_name}'.", is_error=True)
            return

        self._last_dlc_row_idx = None
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        edit_revision = self.begin_edit_revision()
        self._pending_save = True
        self.queue_pending_export(int(self.current_frame), f"CLEAR:{point_name}", edit_revision)
        self.note_unsaved_edit(
            f"Frame {self.current_frame}: cleared '{point_name}' (set to NaN).",
            rerender_preview=True,
        )

    def flip_ear_points_current_frame(self):
        """Swap left/right ear coordinates on the current frame and mark both as manually edited."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return

        ear_pair = self.find_ear_point_pair()
        if ear_pair is None:
            self.set_edit_status("No left/right ear pair found in this DLC file.", is_error=True)
            return

        dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
        if dlc_row_idx is None:
            self.set_edit_status("No DLC row for this frame.", is_error=True)
            return

        left_point, right_point = ear_pair
        swapped_axes = 0
        for suffix in ("_x", "_y"):
            left_col = f"{left_point}{suffix}"
            right_col = f"{right_point}{suffix}"
            if left_col not in self.dlc_data.columns or right_col not in self.dlc_data.columns:
                self.set_edit_status(
                    f"Missing coordinate columns for ear swap ({left_col}, {right_col}).",
                    is_error=True,
                )
                return

            left_val = self.get_cell_value(dlc_row_idx, left_col)
            right_val = self.get_cell_value(dlc_row_idx, right_col)
            self.set_cell_value(dlc_row_idx, left_col, right_val)
            self.set_cell_value(dlc_row_idx, right_col, left_val)
            swapped_axes += 1

        self.ensure_confidence_columns([left_point, right_point])
        conf_updates = self.set_points_confidence_for_rows(
            [int(dlc_row_idx)], [left_point, right_point], self.manual_edit_confidence
        )

        self._last_dlc_row_idx = None
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        edit_revision = self.begin_edit_revision()
        self._pending_save = True
        self.queue_pending_export(int(self.current_frame), [left_point, right_point], edit_revision)
        self.note_unsaved_edit(
            f"Frame {self.current_frame}: swapped {left_point} <-> {right_point} "
            f"(axes={swapped_axes}; conf={self.manual_edit_confidence:.2f}; conf_cols={conf_updates}).",
            rerender_preview=True,
        )

    def flip_ear_points_frame_range(self, start_frame: int, end_frame: int) -> bool:
        """Swap left/right ear coordinates across an inclusive frame range."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return False
        if self.total_frames <= 0:
            self.set_edit_status("Video metadata missing; reload files and try again.", is_error=True)
            return False

        ear_pair = self.find_ear_point_pair()
        if ear_pair is None:
            self.set_edit_status("No left/right ear pair found in this DLC file.", is_error=True)
            return False

        left_point, right_point = ear_pair
        step = 1 if end_frame >= start_frame else -1
        target_frames = list(range(int(start_frame), int(end_frame) + step, step))
        if not target_frames:
            self.set_edit_status("No frames selected for ear flip.", is_error=True)
            return False

        row_to_frames: Dict[int, List[int]] = {}
        skipped_unmapped = 0
        edit_revision = self.begin_edit_revision()
        for frame_num in target_frames:
            dlc_row_idx = self.get_dlc_row_for_frame(int(frame_num))
            if dlc_row_idx is None:
                skipped_unmapped += 1
                continue
            row_to_frames.setdefault(int(dlc_row_idx), []).append(int(frame_num))
            self.queue_pending_export(int(frame_num), [left_point, right_point], edit_revision)

        if not row_to_frames:
            self.set_edit_status("Flip L/R ears range: no frames mapped to a DLC row.", is_error=True)
            return False

        swapped_axes = 0
        for suffix in ("_x", "_y"):
            left_col = f"{left_point}{suffix}"
            right_col = f"{right_point}{suffix}"
            if left_col not in self.dlc_data.columns or right_col not in self.dlc_data.columns:
                self.set_edit_status(
                    f"Missing coordinate columns for ear swap ({left_col}, {right_col}).",
                    is_error=True,
                )
                return False

            unique_rows = sorted(row_to_frames.keys())
            left_vals = self.dlc_data.loc[unique_rows, left_col].copy()
            right_vals = self.dlc_data.loc[unique_rows, right_col].copy()
            self.dlc_data.loc[unique_rows, left_col] = right_vals.to_numpy()
            self.dlc_data.loc[unique_rows, right_col] = left_vals.to_numpy()
            swapped_axes += 1

        self.ensure_confidence_columns([left_point, right_point])
        conf_updates = self.set_points_confidence_for_rows(
            sorted(row_to_frames.keys()), [left_point, right_point], self.manual_edit_confidence
        )

        self._last_dlc_row_idx = None
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        self._pending_save = True

        touched_frames = sum(len(frames) for frames in row_to_frames.values())
        duplicate_row_frames = touched_frames - len(row_to_frames)
        status_bits = [
            f"Flip L/R ears {start_frame}->{end_frame}: swapped {left_point} <-> {right_point}",
            f"frames={touched_frames}/{len(target_frames)}",
            f"rows={len(row_to_frames)}",
            f"axes={swapped_axes}",
            f"conf={self.manual_edit_confidence:.2f}",
            f"conf_cols={conf_updates}",
        ]
        if skipped_unmapped:
            status_bits.append(f"skipped {skipped_unmapped} unmapped frame(s)")
        if duplicate_row_frames:
            status_bits.append(f"{duplicate_row_frames} frame(s) shared a DLC row")

        rerender_preview = any(int(self.current_frame) in frames for frames in row_to_frames.values())
        self.note_unsaved_edit("; ".join(status_bits) + ".", rerender_preview=rerender_preview)
        return True

    def refresh_clear_point_combo(self, point_names: Optional[List[str]] = None):
        """Refresh point dropdown used by per-point clear action."""
        if not hasattr(self, "clear_point_combo"):
            return
        if point_names is None:
            point_names = sorted(self.editable_point_names)

        current_text = self.clear_point_combo.currentText().strip()
        self.clear_point_combo.blockSignals(True)
        self.clear_point_combo.clear()
        if point_names:
            self.clear_point_combo.addItems(point_names)
            if current_text in point_names:
                self.clear_point_combo.setCurrentText(current_text)
            self.clear_point_combo.setEnabled(True)
            self.clear_point_btn.setEnabled(True)
        else:
            self.clear_point_combo.addItem("(no points)")
            self.clear_point_combo.setEnabled(False)
            self.clear_point_btn.setEnabled(False)
        self.clear_point_combo.blockSignals(False)
        self.refresh_flip_ears_button()

    def get_editable_copy_spec(self) -> Tuple[List[int], int]:
        """Return dataframe column indexes to copy for editable points."""
        if self.dlc_data is None:
            return [], 0

        col_indices = []
        seen_cols = set()
        copied_points = 0

        for point_name in self.editable_point_names:
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"

            has_xy = False
            for col_name in (x_col, y_col):
                col_idx = self._column_index(col_name)
                if col_idx is None or col_idx in seen_cols:
                    continue
                col_indices.append(col_idx)
                seen_cols.add(col_idx)
                has_xy = True

            if has_xy:
                copied_points += 1

            for conf_col in self.get_point_confidence_columns(point_name):
                conf_idx = self._column_index(conf_col)
                if conf_idx is None or conf_idx in seen_cols:
                    continue
                col_indices.append(conf_idx)
                seen_cols.add(conf_idx)

        return col_indices, copied_points

    def copy_source_frame_to_target_frames(
        self,
        source_frame: int,
        target_frames: List[int],
        export_tag: str,
        action_label: str,
    ) -> bool:
        """Copy editable points from one source frame into a list of target frames."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return False
        if not self.editable_point_names:
            self.set_edit_status("No editable points found for this file.", is_error=True)
            return False
        if self.total_frames <= 0:
            self.set_edit_status("Video metadata missing; reload files and try again.", is_error=True)
            return False
        if not target_frames:
            self.set_edit_status("No target frames selected.", is_error=True)
            return False

        source_row_idx = self.get_dlc_row_for_frame(int(source_frame))
        if source_row_idx is None:
            self.set_edit_status(f"Could not map source frame {source_frame} to a DLC row.", is_error=True)
            return False

        col_indices, copied_points_per_row = self.get_editable_copy_spec()
        if not col_indices or copied_points_per_row == 0:
            self.set_edit_status("No editable point columns available to copy.", is_error=True)
            return False

        mapped_target_rows = []
        skipped_unmapped = 0
        skipped_same_row = 0

        edit_revision = self.begin_edit_revision()
        for target_frame in target_frames:
            target_row_idx = self.get_dlc_row_for_frame(int(target_frame))
            if target_row_idx is None:
                skipped_unmapped += 1
                continue
            if int(target_row_idx) == int(source_row_idx):
                skipped_same_row += 1
                continue

            mapped_target_rows.append((int(target_frame), int(target_row_idx)))
            self.queue_pending_export(int(target_frame), export_tag, edit_revision)

        if not mapped_target_rows:
            self.set_edit_status(
                f"{action_label}: no target frames mapped to a different DLC row.",
                is_error=True,
            )
            return False

        unique_target_rows = []
        seen_rows = set()
        duplicate_row_targets = 0
        for _, row_idx in mapped_target_rows:
            if row_idx in seen_rows:
                duplicate_row_targets += 1
                continue
            seen_rows.add(row_idx)
            unique_target_rows.append(row_idx)

        source_values = self.dlc_data.iloc[int(source_row_idx), col_indices].to_numpy(copy=True)
        values_matrix = np.tile(source_values, (len(unique_target_rows), 1))
        self.dlc_data.iloc[unique_target_rows, col_indices] = values_matrix
        self.set_points_confidence_for_rows(
            unique_target_rows, self.editable_point_names, self.manual_edit_confidence
        )

        self._last_dlc_row_idx = None
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        self._pending_save = True
        self.render_drag_preview()

        touched_frame_count = len(mapped_target_rows)
        copied_total = copied_points_per_row * len(unique_target_rows)
        status_bits = [
            f"{action_label}: copied frame {source_frame} into {touched_frame_count}/{len(target_frames)} target frame(s)",
            f"updated {len(unique_target_rows)} DLC row(s)",
            f"copied {copied_total} point sets",
            f"conf={self.manual_edit_confidence:.2f}",
        ]
        if skipped_unmapped:
            status_bits.append(f"skipped {skipped_unmapped} unmapped frame(s)")
        if skipped_same_row:
            status_bits.append(f"skipped {skipped_same_row} same-row frame(s)")
        if duplicate_row_targets:
            status_bits.append(f"{duplicate_row_targets} frame(s) shared a DLC row")

        self.note_unsaved_edit("; ".join(status_bits) + ".")
        return True

    def prompt_copy_source_frame_range(self):
        """Ask for source/till frames and copy editable points across that range."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return
        if self.total_frames <= 0:
            self.set_edit_status("Video metadata missing; reload files and try again.", is_error=True)
            return

        min_frame = max(0, int(self.start_frame))
        max_frame = max(min_frame, int(self.total_frames - 1))

        default_source = max(min_frame, min(max_frame, int(self.current_frame)))
        if default_source < max_frame:
            default_till = min(max_frame, default_source + 10)
        else:
            default_till = max(min_frame, default_source - 10)

        dialog = CopyFrameRangeDialog(
            min_frame=min_frame,
            max_frame=max_frame,
            source_frame=default_source,
            till_frame=default_till,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        source_frame, till_frame = dialog.get_selected_frames()
        if source_frame == till_frame:
            self.set_edit_status("Source frame and till frame must be different.", is_error=True)
            return

        step = 1 if till_frame > source_frame else -1
        target_frames = list(range(source_frame + step, till_frame + step, step))
        self.copy_source_frame_to_target_frames(
            source_frame=source_frame,
            target_frames=target_frames,
            export_tag=f"APPLY_RANGE:{source_frame}->{till_frame}",
            action_label=f"Range apply {source_frame}->{till_frame}",
        )

    def prompt_flip_ears_frame_range(self):
        """Ask for an inclusive frame range and swap left/right ears across it."""
        if self.dlc_data is None or self.cap is None:
            self.set_edit_status("Load video + DLC first.", is_error=True)
            return
        if self.total_frames <= 0:
            self.set_edit_status("Video metadata missing; reload files and try again.", is_error=True)
            return

        ear_pair = self.find_ear_point_pair()
        if ear_pair is None:
            self.set_edit_status("No left/right ear pair found in this DLC file.", is_error=True)
            return

        min_frame = max(0, int(self.start_frame))
        max_frame = max(min_frame, int(self.total_frames - 1))
        default_start = max(min_frame, min(max_frame, int(self.current_frame)))
        default_end = min(max_frame, default_start + 10) if default_start < max_frame else default_start

        dialog = FrameRangeDialog(
            title="Flip L/R Ears Across Frames",
            description=(
                f"Swap {ear_pair[0]} and {ear_pair[1]} coordinates on every frame in the inclusive range."
            ),
            start_label="Start frame:",
            end_label="End frame:",
            min_frame=min_frame,
            max_frame=max_frame,
            start_frame=default_start,
            end_frame=default_end,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        start_frame, end_frame = dialog.get_selected_frames()
        self.flip_ear_points_frame_range(start_frame, end_frame)

    def apply_previous_frame_to_current(self):
        """Copy previous frame *_cam values into current frame."""
        prev_frame = self.current_frame - 1
        if prev_frame < 0:
            self.set_edit_status("No previous frame available.", is_error=True)
            return

        self.copy_source_frame_to_target_frames(
            source_frame=prev_frame,
            target_frames=[int(self.current_frame)],
            export_tag="APPLY_PREV",
            action_label="Apply prev -> current",
        )

    def apply_current_frame_to_next_10(self):
        """Copy current frame *_cam values into the next 10 video frames."""
        if self.total_frames <= 0:
            self.set_edit_status("Video metadata missing; reload files and try again.", is_error=True)
            return

        start_target_frame = self.current_frame + 1
        if start_target_frame >= self.total_frames:
            self.set_edit_status("No next frame available.", is_error=True)
            return

        end_target_frame = min(self.current_frame + 10, self.total_frames - 1)

        self.copy_source_frame_to_target_frames(
            source_frame=int(self.current_frame),
            target_frames=list(range(start_target_frame, end_target_frame + 1)),
            export_tag="APPLY_NEXT10",
            action_label=f"Apply current -> next 10 ({start_target_frame}-{end_target_frame})",
        )
    
    def update_video_display(self, frame_rgb: np.ndarray, fast: bool = False):
        """Convert an RGB frame to pixmap and show it."""
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation if fast else Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Cache display geometry for click->frame coordinate mapping in edit mode.
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        disp_w = scaled_pixmap.width()
        disp_h = scaled_pixmap.height()
        x_offset = max(0, (label_w - disp_w) // 2)
        y_offset = max(0, (label_h - disp_h) // 2)
        self._display_rect = (x_offset, y_offset, disp_w, disp_h)
    
    def render_drag_preview(self):
        """Fast redraw path used during drag, using cached frame."""
        if self._last_rendered_rgb_base is None or self.dlc_data is None:
            return
        
        frame_rgb = self._last_rendered_rgb_base.copy()
        frame_h, frame_w = frame_rgb.shape[:2]
        dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
        if dlc_row_idx is None:
            return
        
        row = self.dlc_data.iloc[dlc_row_idx]
        point_coords_text = []
        
        candidate_points = [
            pn for pn in self.point_configs.keys()
            if pn in self.editable_point_names and self.point_configs[pn].get("enabled", True)
        ]
        
        for point_name in candidate_points:
            config = self.point_configs.get(point_name)
            if not config:
                continue
            
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                continue
            
            x_val = row[x_col]
            y_val = row[y_col]
            if (
                self.dragging_dlc_row_idx == dlc_row_idx
                and point_name in self.drag_preview_points_raw
            ):
                x_val, y_val = self.drag_preview_points_raw[point_name]
            elif (
                self.dragging_point_name == point_name
                and self.dragging_dlc_row_idx == dlc_row_idx
                and self.drag_preview_raw is not None
            ):
                x_val, y_val = self.drag_preview_raw
            if pd.isna(x_val) or pd.isna(y_val):
                point_coords_text.append(f"{point_name}: NaN")
                continue

            x_px, y_px = self.point_raw_to_pixels(float(x_val), float(y_val), frame_w, frame_h)
            conf_val = self.get_point_confidence_value_from_data(row, point_name)
            color = config["color"]
            is_dragging_this_point = (
                self.dragging_dlc_row_idx == dlc_row_idx
                and (
                    (
                        (self.drag_all_points or self.drag_rotate_points)
                        and point_name in self.drag_source_points_raw
                    )
                    or (
                        not self.drag_all_points
                        and not self.drag_rotate_points
                        and self.dragging_point_name == point_name
                    )
                )
            )
            dot_radius = 8 if is_dragging_this_point else 5
            ring_radius = 11 if is_dragging_this_point else 7
            ring_color = (255, 220, 0) if is_dragging_this_point else (255, 255, 255)
            cv2.circle(frame_rgb, (x_px, y_px), dot_radius, color, -1)
            cv2.circle(frame_rgb, (x_px, y_px), ring_radius, ring_color, 2 if is_dragging_this_point else 1)
            if conf_val is not None:
                point_coords_text.append(f"{point_name}: ({x_px}, {y_px}) conf={conf_val:.2f}")
            else:
                point_coords_text.append(f"{point_name}: ({x_px}, {y_px})")
        
        self.update_video_display(frame_rgb, fast=True)
        if point_coords_text:
            self.coords_label.setText("Point Coordinates:\n" + "\n".join(point_coords_text))
        else:
            self.coords_label.setText("Point Coordinates: No points visible")
    
    def export_current_edit_label(
        self,
        dlc_row_idx: int,
        edited_point: str,
        frame_num: Optional[int] = None,
        frame_bgr: Optional[np.ndarray] = None
    ):
        """Save a frame image and point labels to manual_labels folder."""
        if self.dlc_data is None:
            return
        if frame_num is None:
            frame_num = int(self.current_frame)
        if frame_bgr is None:
            frame_bgr = self._last_rendered_bgr
        if frame_bgr is None:
            return
        self.init_manual_export_paths()
        if self.export_frames_dir is None or self.export_labels_dir is None or self.export_log_path is None:
            return
        
        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return
        
        video_prefix = self.get_current_video_export_prefix()
        base_name = f"{video_prefix}_f{frame_num:07d}"
        img_path = self.export_frames_dir / f"{base_name}.png"
        label_path = self.export_labels_dir / f"{base_name}.txt"
        
        cv2.imwrite(str(img_path), frame_bgr)
        
        row = self.dlc_data.iloc[dlc_row_idx]
        lines = [
            f"# video_path={self.video_path}",
            f"# dlc_path={self.dlc_path}",
            f"# frame={frame_num}",
            f"# dlc_row={dlc_row_idx}",
            f"# edited_point={edited_point}",
            f"# timestamp={datetime.utcnow().isoformat()}Z",
            "point_name,x_raw,y_raw,x_pixel,y_pixel,x_norm,y_norm,confidence",
        ]
        
        for point_name in sorted(self.editable_point_names):
            x_col = f"{point_name}_x"
            y_col = f"{point_name}_y"
            if x_col not in self.dlc_data.columns or y_col not in self.dlc_data.columns:
                continue
            x_val = row[x_col]
            y_val = row[y_col]
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            
            x_raw = float(x_val)
            y_raw = float(y_val)
            x_px, y_px = self.point_raw_to_pixels(x_raw, y_raw, frame_w, frame_h)
            x_norm = x_raw if self.coords_are_normalized else (x_raw / max(frame_w, 1))
            y_norm = y_raw if self.coords_are_normalized else (y_raw / max(frame_h, 1))
            
            conf_val = None
            for suffix in ["_prob", "_likelihood", "_conf"]:
                conf_col = f"{point_name}{suffix}"
                if conf_col in self.dlc_data.columns and pd.notna(row[conf_col]):
                    conf_val = float(row[conf_col])
                    break
            if conf_val is None and point_name.endswith("_cam"):
                base_name_no_cam = point_name[:-4]
                for suffix in ["_prob", "_likelihood", "_conf"]:
                    conf_col = f"{base_name_no_cam}{suffix}"
                    if conf_col in self.dlc_data.columns and pd.notna(row[conf_col]):
                        conf_val = float(row[conf_col])
                        break
            conf_text = "" if conf_val is None else f"{conf_val:.6f}"
            
            lines.append(
                f"{point_name},{x_raw:.6f},{y_raw:.6f},{x_px},{y_px},{x_norm:.6f},{y_norm:.6f},{conf_text}"
            )
        
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        
        with open(self.export_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat() + "Z",
                self.video_path,
                self.dlc_path,
                frame_num,
                dlc_row_idx,
                edited_point,
                str(img_path),
                str(label_path),
            ])
    
    def read_frame_bgr(self, frame_num: int) -> Optional[np.ndarray]:
        """Read a specific frame from video for export without changing UI state."""
        if self.cap is None:
            return None
        try:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if not ret:
                return None
            return frame
        except Exception:
            return None
    
    def export_pending_frames(self):
        """Export all frames touched in current edit session."""
        if not self.pending_export_frames:
            return
        
        pending = sorted(self.pending_export_frames.items(), key=lambda item: item[0])
        for frame_num, edited_points in pending:
            dlc_row_idx = self.get_dlc_row_for_frame(int(frame_num))
            if dlc_row_idx is None:
                continue
            
            if int(frame_num) == int(self.current_frame) and self._last_rendered_bgr is not None:
                frame_bgr = self._last_rendered_bgr.copy()
            else:
                frame_bgr = self.read_frame_bgr(int(frame_num))
            if frame_bgr is None:
                continue
            
            edited_point_text = "|".join(sorted(edited_points)) if edited_points else "EDITED"
            self.export_current_edit_label(
                dlc_row_idx=dlc_row_idx,
                edited_point=edited_point_text,
                frame_num=int(frame_num),
                frame_bgr=frame_bgr
            )
        
        self.pending_export_frames.clear()
    
    def autosave_edits(self):
        """Autosave is intentionally disabled; keep reminding the user to save explicitly."""
        if self.save_timer.isActive():
            self.save_timer.stop()
        if self._pending_save or self.pending_export_frames:
            self.set_edit_status(
                "Autosave is disabled. Click Save Edits Now to write DLC + exports.",
                is_error=False,
            )
    
    def persist_dlc_edits(self, export_frames: bool = True, autosave: bool = False):
        """Snapshot edits and write DLC data in a background worker."""
        if self.dlc_data is None or not self.dlc_path:
            return
        if self.save_timer.isActive():
            self.save_timer.stop()
        if self.dlc_changed_on_disk():
            if autosave:
                self.set_edit_status(
                    "Autosave paused because the DLC file changed on disk. Reload it before saving again.",
                    is_error=True,
                )
            else:
                self.set_edit_status(
                    "DLC file changed on disk. Reload it before saving to avoid overwriting newer changes.",
                    is_error=True,
                )
            return
        if not self._pending_save and (not self.pending_export_frames or not export_frames):
            if not autosave:
                self.set_edit_status("No pending edits to save.")
            return

        if self._save_in_progress:
            self._save_followup_requested = True
            self._save_followup_export = self._save_followup_export or export_frames
            if not autosave:
                self.set_edit_status("Save already in progress. Queued another save.")
            return

        job = self.build_save_job(export_frames=export_frames, autosave=autosave)
        self.launch_save_worker(job)

    def refresh_background_action_buttons(self):
        """Keep retrain/calibration buttons in sync with background jobs."""
        retrain_running = self.retrain_process is not None and self.retrain_process.poll() is None
        calibration_running = bool(self._calibration_in_progress)
        actions_busy = retrain_running or calibration_running
        if hasattr(self, "retrain_btn"):
            self.retrain_btn.setEnabled(not actions_busy)
        if hasattr(self, "reapply_calibration_btn"):
            self.reapply_calibration_btn.setEnabled(not actions_busy)

    def set_calibration_status(self, message: str, is_error: bool = False):
        """Update calibration workflow status line."""
        color = "#A91E2C" if is_error else "#1F4B99"
        self.calibration_status_label.setStyleSheet(f"font-size: 8.5pt; color: {color};")
        self.calibration_status_label.setText(message)

    def on_calibration_preset_changed(self):
        """Apply hard-coded calibration defaults selected in the UI."""
        preset_key = self.calibration_preset_combo.currentData()
        if not preset_key:
            self.save_preferences()
            return
        preset = CALIBRATION_PRESETS.get(str(preset_key))
        if not preset:
            return
        self.calibration_dir_input.setText(preset["calibration_dir"])
        self.screen_start_x_input.setText(preset["screen_start_x"])
        self.screen_pix_cm_input.setText(preset["screen_pix_cm"])
        self.screen_y_input.setText(preset["screen_y"])
        self.save_preferences()

    def on_calibration_field_edited(self):
        """Manual edits mean the calibration fields no longer exactly represent a preset."""
        if self.calibration_preset_combo.currentData():
            self.calibration_preset_combo.setCurrentIndex(0)
        else:
            self.save_preferences()

    def get_configured_calibration_dir(self) -> Optional[Path]:
        """Return the selected calibration directory, falling back to the app default."""
        text = self.calibration_dir_input.text().strip()
        if text:
            return Path(text).expanduser()
        return self.default_calibration_dir

    def get_configured_screen_calibration(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return screen calibration values used by run_model for bug cm/deviation columns."""
        values = []
        for field in (self.screen_start_x_input, self.screen_pix_cm_input, self.screen_y_input):
            text = field.text().strip()
            values.append(float(text) if text else None)
        return values[0], values[1], values[2]

    def build_calibration_job(self) -> Dict[str, Any]:
        """Capture the current DLC/video context for recalibration."""
        calibration_dir = self.get_configured_calibration_dir()
        screen_start_x, screen_pix_cm, screen_y = self.get_configured_screen_calibration()
        return {
            "pose_file": self.dlc_path,
            "video_path": self.video_path,
            "calibration_dir": str(calibration_dir) if calibration_dir is not None else None,
            "screen_start_x": screen_start_x,
            "screen_pix_cm": screen_pix_cm,
            "screen_y": screen_y,
        }

    def launch_calibration_worker(self, job: Dict[str, Any]):
        """Run calibration helper in a background thread."""
        self._calibration_in_progress = True
        self.refresh_background_action_buttons()

        self._calibration_thread = QThread(self)
        self._calibration_worker = CalibrationWorker(job)
        self._calibration_worker.moveToThread(self._calibration_thread)
        self._calibration_thread.started.connect(self._calibration_worker.run)
        self._calibration_worker.finished.connect(self.on_calibration_worker_finished)
        self._calibration_worker.failed.connect(self.on_calibration_worker_failed)
        self._calibration_worker.finished.connect(self._calibration_thread.quit)
        self._calibration_worker.failed.connect(self._calibration_thread.quit)
        self._calibration_worker.finished.connect(self._calibration_worker.deleteLater)
        self._calibration_worker.failed.connect(self._calibration_worker.deleteLater)
        self._calibration_thread.finished.connect(self._calibration_thread.deleteLater)
        self._calibration_thread.finished.connect(self.on_calibration_thread_finished)
        self._calibration_thread.start()

    def on_calibration_thread_finished(self):
        """Drop worker references once the calibration thread exits."""
        self._calibration_thread = None
        self._calibration_worker = None

    def start_reapply_calibration(self):
        """Save current edits if needed, then reapply calibration to the loaded DLC file."""
        if self._save_in_progress or self._pending_save or self.pending_export_frames:
            self._calibration_after_save = True
            self.set_calibration_status("Saving edits before calibration...", is_error=False)
            self.persist_dlc_edits(export_frames=True, autosave=False)
            return

        if self._calibration_in_progress:
            self.set_calibration_status("Calibration is already running. Wait for completion.", is_error=True)
            return
        if self.retrain_process is not None and self.retrain_process.poll() is None:
            self.set_calibration_status("Retrain is already running. Wait for completion.", is_error=True)
            return
        if self.video_path is None or self.dlc_data is None or not self.dlc_path:
            self.set_calibration_status("Load video + DLC first.", is_error=True)
            return
        if load_calibration_helper() is None:
            self.set_calibration_status(
                f"Calibration helper unavailable: {CALIBRATION_HELPER_IMPORT_ERROR}",
                is_error=True,
            )
            return
        calibration_dir = self.get_configured_calibration_dir()
        if calibration_dir is None:
            self.set_calibration_status("Choose a calibration folder first.", is_error=True)
            return
        if not calibration_dir.exists():
            self.set_calibration_status(f"Calibration folder does not exist: {calibration_dir}", is_error=True)
            return
        try:
            self.get_configured_screen_calibration()
        except ValueError:
            self.set_calibration_status("Screen calibration values must be numbers.", is_error=True)
            return

        self.set_calibration_status(
            f"Reapplying calibration to {Path(self.dlc_path).name} using {calibration_dir}...",
            is_error=False,
        )
        self.save_preferences()
        self.launch_calibration_worker(self.build_calibration_job())

    def on_calibration_worker_finished(self, result: Dict[str, Any]):
        """Handle successful recalibration and reload the current DLC file from disk."""
        self._calibration_in_progress = False
        self.refresh_background_action_buttons()
        updated_cells = int(result.get("updated_cells", 0))
        output_path = result.get("output_path") or self.dlc_path
        log_lines = result.get("log_lines") or []
        if log_lines:
            print("Calibration log:")
            for line in log_lines:
                print(line)
        self.set_calibration_status(
            f"Calibration finished ({updated_cells} calibrated cells).",
            is_error=False,
        )
        self.set_edit_status(
            f"Calibration finished for {output_path}. Reloading DLC from disk...",
            is_error=False,
        )
        if self._close_after_calibration:
            self._close_after_calibration = False
            QTimer.singleShot(0, self.close)
            return
        QTimer.singleShot(0, self.load_files)

    def on_calibration_worker_failed(self, error_text: str):
        """Surface calibration failures without crashing the UI."""
        self._calibration_in_progress = False
        self.refresh_background_action_buttons()
        self._close_after_calibration = False
        summary = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown calibration failure"
        self.set_calibration_status(f"Calibration failed: {summary}", is_error=True)
        self.set_edit_status("Calibration failed. See console output for details.", is_error=True)
        print("Error recalibrating edited DLC file:")
        print(error_text)
    
    def set_retrain_status(self, message: str, is_error: bool = False):
        """Update retrain workflow status line."""
        color = "#A91E2C" if is_error else "#1F4B99"
        self.retrain_status_label.setStyleSheet(f"font-size: 8.5pt; color: {color};")
        self.retrain_status_label.setText(message)
    
    def start_retrain_and_rerun(self):
        """Run external retrain helper script asynchronously, then optional rerun."""
        if self._save_in_progress or self._pending_save or self.pending_export_frames:
            self._retrain_after_save = True
            self.set_retrain_status("Saving edits and exporting labels before retrain...", is_error=False)
            self.persist_dlc_edits(export_frames=True, autosave=False)
            return

        if self._calibration_in_progress:
            self.set_retrain_status("Calibration is already running. Wait for completion.", is_error=True)
            return
        if self.retrain_process is not None and self.retrain_process.poll() is None:
            self.set_retrain_status("Retrain is already running. Wait for completion.", is_error=True)
            return
        if self.video_path is None or self.dlc_data is None:
            self.set_retrain_status("Load video + DLC first.", is_error=True)
            return
        if not self.retrain_helper_script.exists():
            self.set_retrain_status(
                f"Missing helper script: {self.retrain_helper_script}",
                is_error=True
            )
            return
        
        config_path = Path(self.retrain_config_input.text().strip()).expanduser()
        if not config_path.exists():
            self.set_retrain_status("Select a valid DeepLabCut config.yaml.", is_error=True)
            return
        
        run_model_script = self.run_model_script_input.text().strip()
        model_name = self.retrain_model_name_input.text().strip()
        model_path_text = self.retrain_model_path_input.text().strip()
        cam_name = self.retrain_cam_name_input.text().strip() or "top"
        if run_model_script and not Path(run_model_script).expanduser().exists():
            self.set_retrain_status("run_model.py path does not exist.", is_error=True)
            return

        model_path = None
        if model_path_text:
            model_path_candidate = Path(model_path_text).expanduser()
            if model_path_candidate.exists():
                if not model_path_candidate.is_dir():
                    self.set_retrain_status("Model folder path is not a directory.", is_error=True)
                    return
                model_path = model_path_candidate.resolve()
            elif model_path_text != str(self.default_model_root):
                self.set_retrain_status("Model folder path does not exist.", is_error=True)
                return
        
        # Force a full save so both parquet/csv and manual_labels exports are up to date.
        self.persist_dlc_edits(export_frames=True, autosave=False)
        labels_pool_root = self.get_manual_labels_pool_root()
        if labels_pool_root is None:
            self.set_retrain_status("Manual labels folder is not initialized.", is_error=True)
            return

        self.flatten_manual_labels_root(labels_pool_root)
        label_files = sorted((labels_pool_root / "labels" / "train").glob("*.txt"))
        if len(label_files) == 0:
            self.set_retrain_status(
                "No exported label txt files found. Edit points and click Save Edits Now first.",
                is_error=True
            )
            return
        self.set_retrain_status(
            f"Found {len(label_files)} manual label files under {labels_pool_root}. Starting retrain..."
        )
        
        cmd = [
            sys.executable,
            str(self.retrain_helper_script),
            "--dlc-config", str(config_path),
            "--labels-root", str(labels_pool_root),
            "--video-path", str(self.video_path),
            "--iterations", str(int(self.retrain_iters_spin.value())),
            "--cam-name", cam_name,
        ]
        if run_model_script:
            cmd.extend(["--run-model-script", str(Path(run_model_script).expanduser())])
        if model_name:
            cmd.extend(["--model-name", model_name])
        if model_path is not None:
            cmd.extend(["--model-path", str(model_path)])
        
        log_dir = labels_pool_root / "retrain_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.retrain_log_path = log_dir / f"retrain_{timestamp}.log"
        
        try:
            self.retrain_log_handle = open(self.retrain_log_path, "w", encoding="utf-8")
            self.retrain_process = subprocess.Popen(
                cmd,
                stdout=self.retrain_log_handle,
                stderr=subprocess.STDOUT
            )
            self.refresh_background_action_buttons()
            self.retrain_poll_timer.start(1000)
            self.set_retrain_status(
                f"Retrain started (PID {self.retrain_process.pid}). Log: {self.retrain_log_path}"
            )
        except Exception as e:
            if self.retrain_log_handle is not None:
                self.retrain_log_handle.close()
                self.retrain_log_handle = None
            self.retrain_process = None
            self.set_retrain_status(f"Failed to start retrain: {e}", is_error=True)
    
    def poll_retrain_process(self):
        """Check retrain subprocess completion without blocking UI."""
        if self.retrain_process is None:
            self.retrain_poll_timer.stop()
            return
        return_code = self.retrain_process.poll()
        if return_code is None:
            return
        
        self.retrain_poll_timer.stop()
        self.refresh_background_action_buttons()
        if self.retrain_log_handle is not None:
            self.retrain_log_handle.close()
            self.retrain_log_handle = None
        
        log_text = str(self.retrain_log_path) if self.retrain_log_path else "(no log file)"
        if return_code == 0:
            self.set_retrain_status(f"Retrain finished successfully. Log: {log_text}")
            self.set_edit_status(
                f"Retrain finished. Check log: {log_text}"
            )
        else:
            self.set_retrain_status(
                f"Retrain failed with exit code {return_code}. Log: {log_text}",
                is_error=True
            )
            self.set_edit_status(
                f"Retrain failed (exit {return_code}). Check log: {log_text}",
                is_error=True
            )
        
        self.retrain_process = None
        self.refresh_background_action_buttons()
    
    def try_autoload_dlc(self, video_path: str):
        """Try to automatically load DLC file from predictions folder"""
        try:
            video_dir = Path(video_path).parent
            predictions_dir = video_dir / "predictions"
            
            if not predictions_dir.exists():
                print(f"No predictions folder found at {predictions_dir}")
                return
            
            # Look for parquet files in predictions folder
            parquet_files = list(predictions_dir.glob("*.parquet"))
            
            if len(parquet_files) == 1:
                # Exactly one parquet file found - auto-load it
                dlc_path = str(parquet_files[0])
                self.dlc_input.setText(dlc_path)
                self.last_dlc_dir = str(predictions_dir)
                print(f"Auto-loaded DLC file: {dlc_path}")
            elif len(parquet_files) == 0:
                print(f"No parquet files found in {predictions_dir}")
            else:
                print(f"Multiple parquet files found in {predictions_dir}, not auto-loading")
        except Exception as e:
            print(f"Error trying to auto-load DLC: {e}")
    
    def on_start_second_changed(self, value):
        """Handle start second change - rebuild frame map if DLC is loaded"""
        if self.dlc_data is not None and self.cap is not None:
            old_start_frame = self.start_frame
            self.start_second = value
            self.start_frame = int(self.start_second * self.fps)
            print(f"\n=== Start frame changed: {old_start_frame} -> {self.start_frame} ===")
            print("Rebuilding frame map...")
            self.build_frame_map()
            # Update current frame to new start
            self.current_frame = self.start_frame
            self.progress_slider.setMinimum(self.start_frame)
            self.progress_slider.setValue(self.start_frame)
            self.display_frame()
            print("=== Frame map rebuilt ===\n")
            
    def load_files(self):
        """Load video and DLC files"""
        video_path = self.video_input.text().strip()
        dlc_path = self.dlc_input.text().strip()
        has_pending_local_edits = (
            self.save_timer.isActive()
            or self._pending_save
            or bool(self.pending_export_frames)
        )
        current_file_changed = self.dlc_changed_on_disk()
        target_matches_current = self._same_path(dlc_path, self.dlc_path)

        if self._calibration_in_progress:
            self.set_edit_status(
                "Calibration in progress. Wait for it to finish before loading new files.",
                is_error=True,
            )
            return

        if self._save_in_progress:
            self._load_after_save = True
            self._load_after_save_paths = (video_path, dlc_path)
            self.set_edit_status(
                "Save/export in progress. Requested files will load when it finishes.",
                is_error=False,
            )
            return

        if current_file_changed and has_pending_local_edits:
            confirmed = self.confirm_discard_unsaved_reload(
                "The DLC file changed on disk after this window loaded it.\n\n"
                "Reloading now will discard the current unsaved in-memory edits from this window."
            )
            if not confirmed:
                self.set_edit_status("Reload canceled. Current in-memory edits were kept.", is_error=True)
                return
            self.clear_pending_save_state()
            self.set_edit_status("Discarded stale in-memory edits and reloading from disk.", is_error=False)
        elif has_pending_local_edits:
            self.queue_load_after_save(
                video_path,
                dlc_path,
                "Saving current edits before loading requested files...",
            )
            return
        elif current_file_changed and target_matches_current:
            self.set_edit_status("Reloading newer DLC file from disk.", is_error=False)
        
        # Reset editing session state for new files
        self.dragging_point_name = None
        self.dragging_dlc_row_idx = None
        self.drag_start_frame = None
        self.drag_start_frame_pos = None
        self.drag_all_points = False
        self.drag_source_points_raw = {}
        self.drag_preview_raw = None
        self.drag_preview_points_raw = {}
        self.drag_rotate_points = False
        self.drag_rotation_center_point = None
        self.drag_rotation_center_px = None
        self.drag_rotation_start_angle = None
        self.editable_point_names = set()
        self.pending_export_frames = {}
        self.pending_export_frame_revisions = {}
        self.export_root = None
        self.export_frames_dir = None
        self.export_labels_dir = None
        self.export_log_path = None
        self.refresh_clear_point_combo([])
        self._last_rendered_rgb_base = None
        self._pending_save = False
        self._dlc_backup_created = False
        self._edit_revision = 0
        self._save_followup_requested = False
        self._save_followup_export = False
        self._close_after_save = False
        self._retrain_after_save = False
        self._calibration_after_save = False
        self._close_after_calibration = False
        self._load_after_save = False
        self._load_after_save_paths = None
        self._loaded_dlc_signature = None
        self.coords_are_normalized = False
        if hasattr(self, "_coords_checked"):
            delattr(self, "_coords_checked")
        self.refresh_flip_ears_button()
        if self.save_timer.isActive():
            self.save_timer.stop()
        if self.edit_mode_btn.isChecked():
            self.edit_mode_btn.setChecked(False)
        self.edit_mode = False
        self.edit_mode_btn.setText("Edit: OFF")
        self.edit_mode_btn.setStyleSheet("")
        self.video_label.setCursor(Qt.ArrowCursor)
        
        print(f"\n=== Loading files ===")
        print(f"Video path: {video_path}")
        print(f"DLC path: {dlc_path}")
        
        if not video_path or not dlc_path:
            self.frame_info_label.setText("Please select both video and DLC files")
            return
        
        self.frame_info_label.setText("Loading: Opening video...")
        QApplication.processEvents()  # Update UI
            
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.frame_info_label.setText("Error: Could not open video")
            return
            
        self.video_path = video_path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video loaded: {self.total_frames} frames @ {self.fps} fps")
        
        # Calculate start frame
        self.start_second = self.start_second_spin.value()
        self.start_frame = int(self.start_second * self.fps)
        self.current_frame = self.start_frame
        
        print(f"Start: second {self.start_second}, frame {self.start_frame}")
        
        # Load DLC data
        self.frame_info_label.setText("Loading: Reading DLC data...")
        QApplication.processEvents()  # Update UI
        self.dlc_path = dlc_path
        self.dlc_save_path_label.setText(f"DLC file overwritten on save: {self.dlc_path}")
        try:
            if dlc_path.endswith('.parquet'):
                self.dlc_data = pd.read_parquet(dlc_path)
            else:
                self.dlc_data = pd.read_csv(dlc_path)
            
            # Flatten multi-level columns if present (DLC often has multi-level columns)
            if isinstance(self.dlc_data.columns, pd.MultiIndex):
                print("Detected multi-level columns, flattening...")
                self.dlc_data.columns = [
                    col if not isinstance(col, tuple) else "_".join(str(c) for c in col).strip() 
                    for col in self.dlc_data.columns
                ]
            
            print(f"Loaded DLC data: {len(self.dlc_data)} rows, {len(self.dlc_data.columns)} columns")
            print(f"DLC columns: {list(self.dlc_data.columns[:20])}")
            
            # Check for confidence/probability columns
            conf_cols = [col for col in self.dlc_data.columns if '_prob' in col or '_likelihood' in col or '_conf' in col]
            if conf_cols:
                print(f"?? Found confidence columns: {conf_cols[:5]}")
            else:
                print(f"?? No confidence columns found (_prob, _likelihood, or _conf)")
            
            # Check for trial_id column (with or without trailing underscore)
            trial_id_col = None
            if 'trial_id' in self.dlc_data.columns:
                trial_id_col = 'trial_id'
            elif 'trial_id_' in self.dlc_data.columns:
                trial_id_col = 'trial_id_'
            
            if trial_id_col:
                print(f"?? trial_id column found ({trial_id_col}) - will display in frame info")
                self._trial_id_col = trial_id_col  # Cache for faster access
                self._current_trial_id = None  # Initialize to None
            else:
                print(f"Note: No trial_id column found in DLC data")
                self._trial_id_col = None
                self._current_trial_id = None
            
            # Build frame mapping
            self.frame_info_label.setText("Loading: Building frame map...")
            QApplication.processEvents()  # Update UI
            self.build_frame_map()
        except Exception as e:
            print(f"ERROR loading DLC file: {e}")
            import traceback
            traceback.print_exc()
            self.frame_info_label.setText(f"Error loading DLC file: {e}")
            return
            
        # Load annotation file if provided
        annotation_path = self.annotation_input.text()
        if annotation_path:
            try:
                if annotation_path.endswith('.parquet'):
                    self.annotation_data = pd.read_parquet(annotation_path)
                elif annotation_path.endswith('.csv'):
                    self.annotation_data = pd.read_csv(annotation_path)
                elif annotation_path.endswith('.json'):
                    self.annotation_data = pd.read_json(annotation_path)
                print(f"Loaded annotation data: {len(self.annotation_data)} rows")
            except Exception as e:
                print(f"Warning: Could not load annotation file: {e}")
                self.annotation_data = None
        
        # Parse DLC columns and create point configs
        print(f"Calling parse_dlc_points()...")
        self.frame_info_label.setText("Loading: Parsing DLC points...")
        QApplication.processEvents()  # Update UI
        self.parse_dlc_points()
        print(f"Point configs created: {len(self.point_configs)}")
        
        # Prepare export paths for edit workflow (folders are created lazily on first export).
        self.init_manual_export_paths(create_dirs=False)
        if self.export_root is not None:
            self.set_edit_status(
                f"Edit export folder (created on first export): {self.export_root}"
            )
        
        # Setup progress slider
        print(f"Setting up progress slider...")
        self.frame_info_label.setText("Loading: Setting up controls...")
        QApplication.processEvents()  # Update UI
        self.progress_slider.setMinimum(self.start_frame)
        self.progress_slider.setMaximum(self.total_frames - 1)
        self.progress_slider.setValue(self.start_frame)
        
        # Display first frame
        print(f"Displaying first frame...")
        self.frame_info_label.setText("Loading: Rendering first frame...")
        QApplication.processEvents()  # Update UI
        self.display_frame()
        self.remember_loaded_dlc_signature(self.dlc_path)
        print(f"=== Load complete ===\n")
        
    def parse_dlc_points(self):
        """Parse DLC column names to extract point names"""
        print(f"\n=== Parsing DLC points ===")
        print(f"DLC data shape: {self.dlc_data.shape}")
        print(f"All columns: {list(self.dlc_data.columns)}")
        
        # Clear existing point checkboxes
        for i in reversed(range(self.points_layout.count())): 
            widget = self.points_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
            
        # Extract unique point names (remove _x, _y, _likelihood suffixes)
        all_point_names = set()
        for col in self.dlc_data.columns:
            col_str = str(col)
            # Remove common suffixes
            for suffix in ['_x', '_y', '_likelihood']:
                if col_str.endswith(suffix):
                    point_name = col_str[:-len(suffix)]
                    all_point_names.add(point_name)
                    break
        
        # Prefer editing/displaying *_cam points only (raw camera coordinates).
        cam_point_names = sorted(
            pn for pn in all_point_names
            if pn.endswith('_cam')
            and f"{pn}_x" in self.dlc_data.columns
            and f"{pn}_y" in self.dlc_data.columns
        )
        if cam_point_names:
            point_names = cam_point_names
            self.editable_point_names = set(cam_point_names)
            print(f"Detected {len(point_names)} editable _cam_ points: {point_names}")
        else:
            # Fallback for older files that do not include *_cam columns.
            point_names = sorted(all_point_names)
            self.editable_point_names = set(point_names)
            print("Warning: no *_cam points found; falling back to all points for editing.")
            print(f"Detected {len(point_names)} fallback points: {point_names}")
        
        if len(point_names) == 0:
            # Add a message if no points found
            no_points_label = QLabel("No DLC points found. Check column names.")
            no_points_label.setStyleSheet("color: red;")
            self.points_layout.addWidget(no_points_label)
            self.refresh_clear_point_combo([])
            return

        created_conf_cols = self.ensure_confidence_columns(point_names)
        if created_conf_cols > 0:
            print(f"Created {created_conf_cols} missing *_conf columns for editable points.")
        
        # Drop stale point configs from previous files and keep only current points.
        existing_configs = self.point_configs.copy()
        self.point_configs = {}
        for point_name in point_names:
            cfg = existing_configs.get(point_name, {})
            if (not cfg) and point_name.endswith('_cam'):
                base_name = point_name[:-4]  # e.g. nose_cam -> nose
                cfg = existing_configs.get(base_name, {})
            color = cfg.get('color') if isinstance(cfg, dict) else None
            if isinstance(color, list):
                color = tuple(color)
            if not color:
                color = self.generate_random_color()
            self.point_configs[point_name] = {
                # Keep points visible by default when loading a file so manual editing is immediate.
                'enabled': True,
                'color': color
            }
        
        # Create checkbox and color picker for each point
        for point_name in point_names:
            point_widget = QWidget()
            point_layout = QHBoxLayout(point_widget)
            point_layout.setContentsMargins(0, 0, 0, 0)
            
            # Checkbox
            checkbox = QCheckBox(point_name)
            checkbox.setChecked(self.point_configs[point_name]['enabled'])
            checkbox.stateChanged.connect(lambda state, pn=point_name: self.toggle_point(pn, state))
            point_layout.addWidget(checkbox)
            
            # Color button
            color_btn = QPushButton()
            color_btn.setMaximumWidth(30)
            color = self.point_configs[point_name]['color']
            self.set_button_color(color_btn, color)
            color_btn.clicked.connect(lambda checked, pn=point_name, btn=color_btn: self.pick_color(pn, btn))
            point_layout.addWidget(color_btn)
            
            self.points_layout.addWidget(point_widget)

        self.refresh_clear_point_combo(point_names)
                
        self.save_preferences()
        
    def generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a vibrant, distinct color from a predefined palette"""
        # Predefined palette of distinct, vibrant colors
        color_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (255, 0, 128),    # Pink
            (0, 255, 128),    # Spring green
            (128, 255, 0),    # Chartreuse
            (0, 128, 255),    # Sky blue
            (255, 192, 203),  # Light pink
            (173, 255, 47),   # Green yellow
            (255, 165, 0),    # Orange
            (148, 0, 211),    # Dark violet
        ]
        
        # Use existing configs to determine which color to use next
        used_colors = [cfg['color'] for cfg in self.point_configs.values()]
        
        # Find an unused color from palette
        for color in color_palette:
            if color not in used_colors:
                return color
        
        # If all palette colors used, generate random bright color
        return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
    def set_button_color(self, button: QPushButton, color: Tuple[int, int, int]):
        """Set button background color"""
        button.setStyleSheet(f"QPushButton {{ background-color: rgb{color}; }}")
        
    def pick_color(self, point_name: str, button: QPushButton):
        """Open color picker dialog"""
        current_color = self.point_configs[point_name]['color']
        color = QColorDialog.getColor(QColor(*current_color), self)
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.point_configs[point_name]['color'] = new_color
            self.set_button_color(button, new_color)
            self.save_preferences()
            self.display_frame()
            
    def toggle_point(self, point_name: str, state: int):
        """Toggle point visibility"""
        is_enabled = (state == Qt.Checked)
        if point_name in self.point_configs:
            self.point_configs[point_name]['enabled'] = is_enabled
            print(f"Point {point_name} {'enabled' if is_enabled else 'disabled'}")
            self.save_preferences()
            # Force redraw if video is loaded
            if self.cap is not None:
                self.display_frame()
        
    def display_frame(self):
        """Display current frame with overlays"""
        if self.cap is None:
            return
        
        try:
            # Seek to current frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            
            if not ret:
                self.frame_info_label.setText("End of video")
                self.pause_video()  # Just pause, don't call stop_video to avoid recursion
                return
                
            # Keep an unmodified frame for export when manual edits happen.
            self._last_rendered_bgr = frame.copy()
            self._last_rendered_frame_size = (frame.shape[1], frame.shape[0])
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_rendered_rgb_base = frame.copy()
        
            points_drawn = 0
            collect_coords_text = (not self.is_playing) or self.edit_mode
            point_coords_text = [] if collect_coords_text else None
        
            # Draw DLC points
            if self.dlc_data is not None and len(self.point_configs) > 0:
                # Get corresponding DLC row for current video frame
                dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame)
            
                if dlc_row_idx is None:
                    print(f"WARNING: No DLC row mapping for video frame {self.current_frame}")
                    self._current_trial_id = None
                else:
                    # Debug: show frame-to-row mapping periodically (less frequently)
                    if self.current_frame % 300 == 0 or self.current_frame == self.start_frame:
                        print(f"Frame {self.current_frame} -> DLC row {dlc_row_idx}")
                    
                    # Cache row data as dict for faster access (iloc is slow)
                    if not hasattr(self, '_last_dlc_row_idx') or self._last_dlc_row_idx != dlc_row_idx:
                        self._last_dlc_row_idx = dlc_row_idx
                        self._cached_row = self.dlc_data.iloc[dlc_row_idx].to_dict()
                    row = self._cached_row
                    
                    # Extract trial_id if available (using cached column name)
                    trial_id_value = None  # Initialize to None
                    if hasattr(self, '_trial_id_col') and self._trial_id_col:
                        trial_id_value = row[self._trial_id_col]
                        if pd.notna(trial_id_value):
                            # Found a valid trial_id - update it
                            trial_id_value = int(trial_id_value) if isinstance(trial_id_value, (int, float)) else str(trial_id_value)
                            if not hasattr(self, '_current_trial_id') or trial_id_value != self._current_trial_id:
                                print(f"Frame {self.current_frame}: Trial ID = {trial_id_value}")
                            self._current_trial_id = trial_id_value
                        # else: NaN value - keep displaying the last valid _current_trial_id (don't update it)
                    
                    for point_name, config in self.point_configs.items():
                        # Check if point is enabled
                        if not config.get('enabled', True):
                            if self.current_frame == self.start_frame:
                                print(f"Point {point_name} is disabled, skipping")
                            continue
                            
                        x_col = f"{point_name}_x"
                        y_col = f"{point_name}_y"
                        if x_col not in row or y_col not in row:
                            continue
                        x = row[x_col]
                        y = row[y_col]
                        
                        if pd.notna(x) and pd.notna(y):
                            try:
                                x_raw = float(x)
                                y_raw = float(y)
                                conf_value = self.get_point_confidence_value_from_data(row, point_name)
                                
                                # Auto-detect if coordinates are normalized (0-1 range)
                                if not hasattr(self, '_coords_checked'):
                                    self._coords_checked = True
                                    # Check if values are in 0-1 range (normalized)
                                    if 0 <= x_raw <= 1 and 0 <= y_raw <= 1:
                                        self.coords_are_normalized = True
                                        print(f"Detected normalized coordinates (0-1 range)")
                                    else:
                                        self.coords_are_normalized = False
                                        print(f"Detected pixel coordinates (range: x={x_raw:.1f}, y={y_raw:.1f})")
                                
                                # Convert normalized coordinates to pixels
                                if self.coords_are_normalized:
                                    x_pixel = int(x_raw * frame.shape[1])
                                    y_pixel = int(y_raw * frame.shape[0])
                                else:
                                    x_pixel = int(x_raw)
                                    y_pixel = int(y_raw)
                                
                                # Use coordinates as-is - DLC and OpenCV both use top-left origin (0,0)
                                x_display = x_pixel
                                y_display = y_pixel
                                
                                # Print coordinate values for debugging (less frequently)
                                if self.current_frame % 300 == 0:  # Print every 300 frames to avoid spam
                                    trial_info = f"trial_id={trial_id_value} | " if trial_id_value is not None else ""
                                    print(f"Frame {self.current_frame} | {trial_info}{point_name}: raw=({x_raw:.3f}, {y_raw:.3f}) -> pixel=({x_pixel}, {y_pixel}) -> display=({x_display}, {y_display})")
                                
                                # Check if coordinates are within frame bounds
                                if 0 <= x_display < frame.shape[1] and 0 <= y_display < frame.shape[0]:
                                    color = config['color']
                                    is_dragging_this_point = (
                                        self.edit_mode
                                        and self.dragging_point_name == point_name
                                        and self.dragging_dlc_row_idx == dlc_row_idx
                                    )
                                    dot_radius = 8 if is_dragging_this_point else 5
                                    ring_radius = 11 if is_dragging_this_point else 7
                                    ring_color = (255, 220, 0) if is_dragging_this_point else (255, 255, 255)
                                    cv2.circle(frame, (x_display, y_display), dot_radius, color, -1)
                                    cv2.circle(frame, (x_display, y_display), ring_radius, ring_color, 2 if is_dragging_this_point else 1)
                                    if self.draw_point_names:
                                        cv2.putText(
                                            frame, point_name, (x_display + 10, y_display - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                                        )
                                    points_drawn += 1
                                    # Collect coordinates for display
                                    if collect_coords_text:
                                        if conf_value is not None:
                                            point_coords_text.append(f"{point_name}: ({x_pixel}, {y_pixel}) conf={conf_value:.2f}")
                                        else:
                                            point_coords_text.append(f"{point_name}: ({x_pixel}, {y_pixel})")
                                else:
                                    if self.current_frame % 300 == 0:  # Log every 300 frames
                                        print(f"Point {point_name} out of bounds: ({x}, {y}) vs frame size {frame.shape}")
                                    if collect_coords_text:
                                        point_coords_text.append(f"{point_name}: OUT OF BOUNDS")
                            except (ValueError, TypeError) as e:
                                print(f"Error converting coordinates for {point_name}: {e}")
                        else:
                            if self.current_frame % 300 == 0:
                                print(f"Point {point_name} has NaN values: x={x}, y={y}")
                            if collect_coords_text:
                                point_coords_text.append(f"{point_name}: NaN")
                
                if self.current_frame % 300 == 0:  # Log every 300 frames
                    print(f"Frame {self.current_frame}: Drew {points_drawn}/{len(self.point_configs)} points")
                        
            # Draw annotations if available
            if self.annotation_data is not None:
                self.draw_annotations(frame)
                
            self.update_video_display(frame, fast=(self.edit_mode or self.is_playing))
            
            # Update frame info with trial_id if available
            time_sec = self.current_frame / self.fps if self.fps > 0 else 0
            dlc_row_idx = self.get_dlc_row_for_frame(self.current_frame) if self.dlc_data is not None else None
            dlc_info = f"DLC: {dlc_row_idx}" if dlc_row_idx is not None else "DLC: -"
        
            # Add trial_id to frame info if available
            trial_info = ""
            if hasattr(self, '_trial_id_col') and self._trial_id_col:
                # Trial ID column exists - show it even if None/NaN
                if hasattr(self, '_current_trial_id') and self._current_trial_id is not None:
                    trial_info = f" | Trial: {self._current_trial_id}"
                else:
                    trial_info = " | Trial: N/A"
                
                # Check if this is the last frame of the current trial (only check every 10 frames for performance)
                if self._current_trial_id is not None and self.current_frame % 10 == 0:
                    try:
                        if self.dlc_data is not None and dlc_row_idx is not None and hasattr(self, '_trial_id_col') and self._trial_id_col:
                            current_trial_id = self._current_trial_id
                            # Look ahead to see if next row has different trial_id
                            if dlc_row_idx + 1 < len(self.dlc_data):
                                # Use .iat for faster scalar access
                                next_trial_id = self.dlc_data[self._trial_id_col].iat[dlc_row_idx + 1]
                                if pd.notna(next_trial_id):
                                    next_trial_id = int(next_trial_id) if isinstance(next_trial_id, (int, float)) else str(next_trial_id)
                                    if next_trial_id != current_trial_id:
                                        trial_info += " [TRIAL ENDED]"
                            else:
                                # Last row in entire dataset
                                trial_info += " [TRIAL ENDED]"
                    except Exception as e:
                        # Silently ignore errors in trial end detection to prevent crashes
                        pass
            
            self.frame_info_label.setText(
                f"Frame: {self.current_frame}/{self.total_frames} | "
                f"{dlc_info}/{len(self.dlc_data) if self.dlc_data is not None else 0} | "
                f"Time: {time_sec:.2f}s | FPS: {self.fps:.1f}"
                f"{trial_info}"
            )
            
            # Update point coordinates display
            if collect_coords_text:
                if point_coords_text:
                    self.coords_label.setText("Point Coordinates:\n" + "\n".join(point_coords_text))
                else:
                    self.coords_label.setText("Point Coordinates: No points visible")
            else:
                playing_msg = "Point Coordinates: playing (pause for per-point details)"
                if self.coords_label.text() != playing_msg:
                    self.coords_label.setText(playing_msg)
            
            # Update slider without triggering signal
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(self.current_frame)
            self.progress_slider.blockSignals(False)
        
        except Exception as e:
            print(f"ERROR in display_frame: {e}")
            import traceback
            traceback.print_exc()
            self.frame_info_label.setText(f"Error displaying frame: {e}")
        
    def draw_annotations(self, frame):
        """Draw annotations on frame"""
        if self.annotation_data is None:
            return
            
        # Try to match current frame with annotation data
        # Look for timestamp, frame, or frame_id columns
        time_col = None
        for col in ['timestamp', 'frame', 'frame_id', 'time']:
            if col in self.annotation_data.columns:
                time_col = col
                break
                
        if time_col is None:
            return
            
        # Find annotations for current frame
        current_time = self.current_frame / self.fps if self.fps > 0 else self.current_frame
        
        # Filter annotations near current frame/time
        if 'frame' in self.annotation_data.columns or 'frame_id' in self.annotation_data.columns:
            # Frame-based matching
            matches = self.annotation_data[
                (self.annotation_data[time_col] >= self.current_frame - 1) &
                (self.annotation_data[time_col] <= self.current_frame + 1)
            ]
        else:
            # Time-based matching
            matches = self.annotation_data[
                (self.annotation_data[time_col] >= current_time - 0.1) &
                (self.annotation_data[time_col] <= current_time + 0.1)
            ]
            
        # Draw annotations
        y_offset = 30
        for _, row in matches.iterrows():
            for col in self.annotation_data.columns:
                if col == time_col:
                    continue
                    
                value = row[col]
                if pd.notna(value):
                    # Check if it's a coordinate-like column
                    col_lower = str(col).lower()
                    if '_x' in col_lower or '_y' in col_lower:
                        # Skip coordinates - they're handled by DLC
                        continue
                    
                    # Draw text annotation in top right
                    text = f"{col}: {value}"
                    cv2.putText(
                        frame, text, (frame.shape[1] - 300, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
                    y_offset += 25
                    
            # Special handling for trial_id
            if 'trial_id' in self.annotation_data.columns:
                trial_id = row.get('trial_id')
                if pd.notna(trial_id):
                    text = f"Trial: {trial_id}"
                    cv2.putText(
                        frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3
                    )
                    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
            
    def play_video(self):
        """Start video playback"""
        if self.cap is None:
            return
        if self.edit_mode:
            self.set_edit_status("Turn Edit mode OFF to play video.")
            return
            
        self.is_playing = True
        self.play_btn.setText("Pause")
        
        # Calculate timer interval based on FPS and playback speed
        interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
        self.timer.start(interval)
        
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_btn.setText("Play")
        self.timer.stop()
        
    def stop_video(self):
        """Stop video and return to start"""
        self.pause_video()
        self.clear_drag_state()
        self.current_frame = self.start_frame
        # Update slider and display frame directly
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(self.current_frame)
        self.progress_slider.blockSignals(False)
        self.display_frame()
        
    def next_frame(self):
        """Advance to next frame"""
        if self.current_frame < self.total_frames - 1:
            self.clear_drag_state()
            self.current_frame += 1
            self.display_frame()
        else:
            self.stop_video()
            
    def skip_frames(self, num_frames: int):
        """Skip forward or backward by num_frames"""
        new_frame = self.current_frame + num_frames
        new_frame = max(self.start_frame, min(new_frame, self.total_frames - 1))
        self.clear_drag_state()
        self.current_frame = new_frame
        self.display_frame()
        
    def seek_frame(self, frame_num: int):
        """Seek to specific frame"""
        if not self.is_playing:  # Only seek when paused
            self.clear_drag_state()
            self.current_frame = frame_num
            self.display_frame()
    
    def should_ignore_saved_path(self, path_str: str) -> bool:
        """Ignore temporary test paths when restoring/saving preferences."""
        if not path_str:
            return False
        p = str(Path(path_str).expanduser())
        if p.startswith("/tmp/") or p.startswith("/private/tmp/"):
            return True
        if p.startswith("/var/folders/") and "/T/" in p:
            return True
        if p.startswith("/private/var/folders/") and "/T/" in p:
            return True
        return False
            
    def load_preferences(self):
        """Load saved preferences"""
        if self.prefs_file.exists():
            try:
                with open(self.prefs_file, 'r') as f:
                    prefs = json.load(f)
                    self.point_configs = prefs.get('point_configs', {})
                    
                    # Restore last used files
                    last_video = prefs.get('last_video_path', '')
                    last_dlc = prefs.get('last_dlc_path', '')
                    last_annotation = prefs.get('last_annotation_path', '')
                    last_start_second = prefs.get('last_start_second', 0)
                    
                    # Restore last browse directories
                    saved_video_dir = prefs.get('last_video_dir', str(Path.home()))
                    saved_dlc_dir = prefs.get('last_dlc_dir', str(Path.home()))
                    self.last_video_dir = saved_video_dir if Path(saved_video_dir).exists() else str(Path.home())
                    self.last_dlc_dir = saved_dlc_dir if Path(saved_dlc_dir).exists() else str(Path.home())
                    
                    if last_video and not self.should_ignore_saved_path(last_video) and Path(last_video).exists():
                        self.video_input.setText(last_video)
                    if last_dlc and not self.should_ignore_saved_path(last_dlc) and Path(last_dlc).exists():
                        self.dlc_input.setText(last_dlc)
                    if last_annotation and not self.should_ignore_saved_path(last_annotation) and Path(last_annotation).exists():
                        self.annotation_input.setText(last_annotation)
                    if last_start_second:
                        self.start_second_spin.setValue(last_start_second)
                    
                    retrain_config = prefs.get('retrain_config_path', '')
                    retrain_run_model = prefs.get('retrain_run_model_script', '')
                    manual_labels_root = prefs.get('manual_labels_root', '')
                    retrain_model_path = prefs.get('retrain_model_path', '')
                    retrain_model_name = prefs.get('retrain_model_name', '')
                    retrain_cam_name = prefs.get('retrain_cam_name', 'top')
                    retrain_iterations = prefs.get('retrain_iterations', 5000)
                    calibration_preset = prefs.get('calibration_preset', '')
                    calibration_dir = prefs.get('calibration_dir', '')
                    screen_start_x = prefs.get('screen_start_x', '')
                    screen_pix_cm = prefs.get('screen_pix_cm', '')
                    screen_y = prefs.get('screen_y', '')
                    
                    if retrain_config and Path(retrain_config).exists():
                        self.retrain_config_input.setText(retrain_config)
                    if retrain_run_model and Path(retrain_run_model).exists():
                        self.run_model_script_input.setText(retrain_run_model)
                    elif not self.run_model_script_input.text().strip():
                        # Sensible default if user did not set it yet.
                        default_run_model = Path.home() / "Dev" / "PreyTouch" / "Arena" / "run_model.py"
                        if default_run_model.exists():
                            self.run_model_script_input.setText(str(default_run_model))
                    if manual_labels_root:
                        self.manual_labels_root_input.setText(manual_labels_root)
                    if retrain_model_path:
                        self.retrain_model_path_input.setText(retrain_model_path)
                    if retrain_model_name:
                        self.retrain_model_name_input.setText(retrain_model_name)
                    if retrain_cam_name:
                        self.retrain_cam_name_input.setText(retrain_cam_name)
                    if calibration_dir:
                        self.calibration_dir_input.setText(calibration_dir)
                    if screen_start_x != '':
                        self.screen_start_x_input.setText(str(screen_start_x))
                    if screen_pix_cm != '':
                        self.screen_pix_cm_input.setText(str(screen_pix_cm))
                    if screen_y != '':
                        self.screen_y_input.setText(str(screen_y))
                    if calibration_preset:
                        idx = self.calibration_preset_combo.findData(calibration_preset)
                        if idx >= 0:
                            self.calibration_preset_combo.setCurrentIndex(idx)
                    self.retrain_iters_spin.setValue(
                        max(self.retrain_iters_spin.minimum(), min(self.retrain_iters_spin.maximum(), int(retrain_iterations)))
                    )
            except Exception as e:
                print(f"Could not load preferences: {e}")
        if not self.run_model_script_input.text().strip():
            default_run_model = Path.home() / "Dev" / "PreyTouch" / "Arena" / "run_model.py"
            if default_run_model.exists():
                self.run_model_script_input.setText(str(default_run_model))
        if not self.retrain_model_path_input.text().strip():
            self.retrain_model_path_input.setText(str(self.default_model_root))
        if not self.manual_labels_root_input.text().strip():
            self.manual_labels_root_input.setText(str(self.default_manual_labels_root))
        if not self.calibration_dir_input.text().strip() and self.default_calibration_dir is not None:
            self.calibration_dir_input.setText(str(self.default_calibration_dir))
                
    def save_preferences(self):
        """Save current preferences"""
        try:
            video_path = self.video_input.text().strip()
            dlc_path = self.dlc_input.text().strip()
            annotation_path = self.annotation_input.text().strip()
            
            if self.should_ignore_saved_path(video_path):
                video_path = ""
            if self.should_ignore_saved_path(dlc_path):
                dlc_path = ""
            if self.should_ignore_saved_path(annotation_path):
                annotation_path = ""
            
            prefs = {
                'point_configs': self.point_configs,
                'last_video_path': video_path,
                'last_dlc_path': dlc_path,
                'last_annotation_path': annotation_path,
                'last_start_second': self.start_second_spin.value(),
                'last_video_dir': self.last_video_dir,
                'last_dlc_dir': self.last_dlc_dir,
                'retrain_config_path': self.retrain_config_input.text().strip(),
                'retrain_run_model_script': self.run_model_script_input.text().strip(),
                'manual_labels_root': self.manual_labels_root_input.text().strip(),
                'retrain_model_path': self.retrain_model_path_input.text().strip(),
                'retrain_model_name': self.retrain_model_name_input.text().strip(),
                'retrain_cam_name': self.retrain_cam_name_input.text().strip(),
                'retrain_iterations': int(self.retrain_iters_spin.value()),
                'calibration_preset': self.calibration_preset_combo.currentData() or '',
                'calibration_dir': self.calibration_dir_input.text().strip(),
                'screen_start_x': self.screen_start_x_input.text().strip(),
                'screen_pix_cm': self.screen_pix_cm_input.text().strip(),
                'screen_y': self.screen_y_input.text().strip(),
            }
            with open(self.prefs_file, 'w') as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            print(f"Could not save preferences: {e}")
            
    def build_frame_map(self):
        """Build mapping between video frames and DLC rows"""
        if self.dlc_data is None:
            return
            
        print("\n=== Building frame map ===")
        print(f"Start frame: {self.start_frame}, Start second: {self.start_second}")
        
        # Check if DLC has frame or time columns
        frame_col = None
        time_col = None
        
        for col in self.dlc_data.columns:
            col_lower = str(col).lower()
            if col_lower in ['frame', 'frame_id', 'frame_number', 'frames']:
                frame_col = col
                print(f"Found frame column: {frame_col}")
                break
            elif col_lower in ['time', 'timestamp', 'time_', 'frame_time']:
                time_col = col
                print(f"Found time column: {time_col}")
        
        if frame_col:
            # Direct frame mapping - offset by start_frame (vectorized)
            print("Using direct frame mapping with start_frame offset")
            dlc_frames = self.dlc_data[frame_col].values
            video_frames = dlc_frames + self.start_frame
            self.dlc_frame_map = dict(zip(video_frames, range(len(dlc_frames))))
            print(f"Frame map created: {len(self.dlc_frame_map)} entries")
            if self.dlc_frame_map:
                frame_range = (min(self.dlc_frame_map.keys()), max(self.dlc_frame_map.keys()))
                print(f"Video frame range mapped: {frame_range}")
        elif time_col:
            # Time-based mapping - check if time values look like Unix timestamps or seconds
            dlc_times = self.dlc_data[time_col]
            first_time = dlc_times.iloc[0] if len(dlc_times) > 0 else 0
            
            # If time values are very large (Unix timestamps), use simple 1:1 mapping
            if first_time > 1e9:  # Likely Unix timestamp
                print(f"Detected Unix timestamps (first value: {first_time})")
                print("Using 1:1 frame-to-row mapping")
                
                # Simple 1:1 mapping: video frame N -> DLC row N
                num_rows = len(self.dlc_data)
                self.dlc_frame_map = {i: i for i in range(num_rows)}
                    
                print(f"Frame map created: {len(self.dlc_frame_map)} entries (1:1 mapping)")
                print(f"DLC covers video frames 0 to {num_rows-1}")
                print("Note: Frame-to-row lookup will use direct index (frame N -> row N)")
                if self.dlc_frame_map:
                    frame_range = (min(self.dlc_frame_map.keys()), max(self.dlc_frame_map.keys()))
                    print(f"Video frame range mapped: {frame_range}")
            else:
                # Time values are in seconds - first DLC row time corresponds to start_frame
                print("Using time-based mapping (seconds from DLC start)")
                print(f"First DLC time: {first_time}, maps to video frame {self.start_frame}")
                start_dlc_time = first_time
                self.dlc_frame_map = {}
                for idx in range(len(dlc_times)):
                    time_val = dlc_times.iloc[idx]
                    if pd.notna(time_val):
                        # Time relative to first DLC row
                        time_diff = time_val - start_dlc_time
                        frame_num = self.start_frame + int(time_diff * self.fps)
                        self.dlc_frame_map[frame_num] = idx
                print(f"Frame map created: {len(self.dlc_frame_map)} entries")
        else:
            # Assume 1:1 mapping: first DLC row corresponds to start_frame
            print(f"Using sequential 1:1 mapping: DLC row 0 -> video frame {self.start_frame}")
            self.dlc_frame_map = {self.start_frame + i: i for i in range(len(self.dlc_data))}
            print(f"Frame map created: {len(self.dlc_frame_map)} entries")
        
        print(f"DLC row range: (0, {len(self.dlc_data)-1})")
        if self.dlc_frame_map:
            print(f"Video frames mapped: ({min(self.dlc_frame_map.keys())}, {max(self.dlc_frame_map.keys())})")
            # Show first few mappings for debugging
            sample_keys = sorted(list(self.dlc_frame_map.keys()))[:5]
            print(f"Sample mappings (first 5):")
            for frame in sample_keys:
                print(f"  Video frame {frame} -> DLC row {self.dlc_frame_map[frame]}")
    
    def get_dlc_row_for_frame(self, video_frame: int) -> Optional[int]:
        """Get DLC row index for a given video frame"""
        if self.dlc_data is None:
            return None
        
        # Preferred path: explicit frame map built from frame/time columns.
        if self.dlc_frame_map is not None and video_frame in self.dlc_frame_map:
            return int(self.dlc_frame_map[video_frame])
        
        # Fallback path for sequential DLC rows starting at start_frame.
        row_from_start = video_frame - self.start_frame
        if 0 <= row_from_start < len(self.dlc_data):
            return int(row_from_start)
        
        # Final fallback for datasets that start at frame 0.
        if 0 <= video_frame < len(self.dlc_data):
            return int(video_frame)
        
        return None
    
    def closeEvent(self, event):
        """Clean up on close"""
        if self.save_timer.isActive():
            self.save_timer.stop()
        if self._calibration_in_progress:
            self._close_after_calibration = True
            self.set_calibration_status("Calibration in progress. Window will close when it finishes.")
            event.ignore()
            return
        if self._save_in_progress:
            self._close_after_save = True
            self.set_edit_status("Save/export in progress. Window will close when it finishes.")
            event.ignore()
            return
        if self._pending_save or self.pending_export_frames:
            self._close_after_save = True
            self.persist_dlc_edits(export_frames=True, autosave=False)
            self.set_edit_status("Saving pending edits before closing...")
            event.ignore()
            return
        if self.retrain_poll_timer.isActive():
            self.retrain_poll_timer.stop()
        if self.retrain_process is not None and self.retrain_process.poll() is None:
            try:
                self.retrain_process.terminate()
            except Exception:
                pass
        self.save_preferences()
        if self.retrain_log_handle is not None:
            self.retrain_log_handle.close()
            self.retrain_log_handle = None
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    player = VideoOverlayPlayer()
    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
