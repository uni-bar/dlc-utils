import argparse
import json
import math
import pickle
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


FRAMES_TIMESTAMPS_DIR = "frames_timestamps"
DATE_FORMAT = "%Y%m%dT%H%M%S"


class MissingFile(Exception):
    pass


class CalibrationError(Exception):
    pass


def clean_model_name(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).name).strip("_")


def parse_bodyparts(text):
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def find_yaml_value(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        for value in obj.values():
            found = find_yaml_value(value, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_yaml_value(value, keys)
            if found:
                return found
    return None


def load_bodyparts(model_path, bodyparts_text=None):
    bodyparts = parse_bodyparts(bodyparts_text)
    if bodyparts:
        return bodyparts

    model_path = Path(model_path)
    yaml_paths = []
    if model_path.is_file() and model_path.suffix in {".yaml", ".yml"}:
        yaml_paths.append(model_path)
    elif model_path.is_dir():
        yaml_paths.extend(model_path.rglob("pose_cfg.yaml"))
        yaml_paths.extend(model_path.rglob("config.yaml"))
        yaml_paths.extend(model_path.rglob("*.yaml"))
        yaml_paths.extend(model_path.rglob("*.yml"))

    for yaml_path in yaml_paths:
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        found = find_yaml_value(data, {"bodyparts", "all_joints_names"})
        if found:
            return [str(x) for x in found]

    raise MissingFile("Could not infer bodyparts from model YAML. Pass --bodyparts nose,left_ear,right_ear")


class DLCPredictor:
    def __init__(self, model_path, bodyparts, model_name=None, threshold=0.5):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from dlclive import DLCLive, Processor

        self.model_path = str(Path(model_path).expanduser())
        self.model_name = model_name or clean_model_name(self.model_path)
        self.threshold = threshold
        self.bodyparts = list(bodyparts)
        if "left_ear" in self.bodyparts and "right_ear" in self.bodyparts and "mid_ears" not in self.bodyparts:
            self.bodyparts.append("mid_ears")
        self.detector = DLCLive(self.model_path, processor=Processor())
        self.is_initialized = False

    @property
    def model_bodyparts(self):
        return [bp for bp in self.bodyparts if bp != "mid_ears"]

    def init(self, img):
        if self.is_initialized:
            return
        self.detector.init_inference(img)
        self.is_initialized = True

    def predict(self, img, frame_id):
        self.init(img)
        pred = self.detector.get_pose(img)
        return self.create_pred_df(pred, frame_id)

    def create_pred_df(self, pred, frame_id):
        cols = ["cam_x", "cam_y", "prob"]
        zf = pd.DataFrame(pred, index=self.model_bodyparts, columns=cols)
        if "mid_ears" in self.bodyparts:
            zf.loc["mid_ears", :] = zf.loc[["left_ear", "right_ear"], :].mean()
        zf = zf.loc[self.bodyparts, :]

        row = pd.DataFrame(pd.concat([zf[c] for c in cols]), columns=[frame_id]).T
        row.columns = pd.MultiIndex.from_product([cols, zf.index]).swaplevel(0, 1)
        row.sort_index(axis=1, level=0, inplace=True)
        return row


class Kalman:
    def __init__(self, dt):
        self.is_initiated = False
        self.x = np.zeros(6, dtype=float)
        self.P = np.eye(6)
        self.F = np.array([
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.Q = np.diag([0.01] * 6)
        self.R = np.diag([0.1, 0.1])

    def init(self, x0, y0):
        if pd.isna(x0) or pd.isna(y0):
            x0, y0 = 0, 0
        self.x = np.array([x0, y0, 0, 0, 0, 0], dtype=float)
        self.is_initiated = True

    def get_filtered(self, x, y):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if not pd.isna(x) and not pd.isna(y):
            z = np.array([x, y], dtype=float)
            residual = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ residual
            self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x


def get_last_artifact(search_folder, cache_prefix, image_date=None):
    if image_date and isinstance(image_date, str):
        image_date = datetime.strptime(image_date, DATE_FORMAT)

    last_date, last_path = None, None
    for path in Path(search_folder).glob(f"{cache_prefix}*.pkl"):
        artifact_date = datetime.strptime(path.stem.split("_")[-1], DATE_FORMAT)
        if image_date and artifact_date > image_date:
            continue
        if last_date is None or artifact_date > last_date:
            last_date = artifact_date
            last_path = path
    return last_path


class CharucoCalibration:
    def __init__(self, calib_dir, cam_name):
        self.calib_dir = Path(calib_dir).expanduser()
        self.cam_name = cam_name
        self.undistort_dir = self.calib_dir / "undistortion" / cam_name
        self.real_world_dir = self.calib_dir / "real_world" / cam_name
        self.cache_prefix = f"undistortion_{json.dumps(None)}"
        self.transform_prefix = f"aruco_transformation_{json.dumps(None)}"
        self.mtx = None
        self.dist = None
        self.new_camera_mtx = None
        self.homography = None

    def init(self, image_date=None):
        undistort_path = get_last_artifact(self.undistort_dir, self.cache_prefix, image_date)
        transform_path = get_last_artifact(self.real_world_dir, self.transform_prefix, image_date)
        if not undistort_path or not transform_path:
            raise CalibrationError(f"Missing calibration artifacts for {self.cam_name} in {self.calib_dir}")

        with undistort_path.open("rb") as f:
            params = pickle.load(f)
        with transform_path.open("rb") as f:
            transform = pickle.load(f)

        self.mtx = params["mtx"]
        self.dist = params["dist"]
        w, h = params["w"], params["h"]
        self.new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        self.homography = transform["homography"]

    def get_location(self, frame_x, frame_y):
        uv = cv2.undistortPoints(
            np.array([frame_x, frame_y]).astype("float32"),
            self.mtx,
            self.dist,
            None,
            self.new_camera_mtx,
        ).ravel()
        uv_1 = np.array([[uv[0], uv[1], 1]], dtype=np.float32).T
        result = np.dot(self.homography, uv_1).ravel()
        return result[0] / result[2], result[1] / result[2]


class PoseRunner:
    def __init__(self, predictor, cam_name, calib_dir=None, screen_start_x=None, screen_pix_cm=None, screen_y=None):
        self.predictor = predictor
        self.cam_name = cam_name
        self.calib_dir = calib_dir
        self.caliber = None
        self.commit_bodypart = "mid_ears" if "mid_ears" in predictor.bodyparts else predictor.bodyparts[0]
        self.kinematic_cols = ["x", "y", "vx", "vy", "ax", "ay"]
        self.kalman = None
        self.is_initialized = False
        self.screen_start_x = screen_start_x
        self.screen_pix_cm = screen_pix_cm
        self.screen_y = screen_y

    @property
    def is_screen_configured(self):
        return self.screen_start_x is not None and self.screen_pix_cm is not None and self.screen_y is not None

    def init(self, frame, video_path):
        self.predictor.init(frame)
        if self.calib_dir:
            self.caliber = CharucoCalibration(self.calib_dir, self.cam_name)
            self.caliber.init(parse_video_date(video_path))
        self.is_initialized = True

    def predict_video(self, video_path, skip_existing=True):
        video_path = Path(video_path)
        cache_path = self.get_predicted_cache_path(video_path)
        if skip_existing and cache_path.exists():
            print(f"Skipping existing prediction: {cache_path}")
            return cache_path

        frames_times = load_frames_times(video_path)
        bug_traj = load_bug_trajectory(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            cap.release()
            return None

        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_frames <= 0:
            print(f"Skipping zero-frame video: {video_path}")
            cap.release()
            return None

        n_frames = min(video_frames, len(frames_times))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.kalman = Kalman(dt=1 / fps)

        rows = []
        try:
            for frame_id in tqdm(range(n_frames), desc=video_path.stem):
                ok, frame = cap.read()
                if not ok:
                    break
                if not self.is_initialized:
                    self.init(frame, video_path)
                timestamp = frames_times.loc[frame_id, "time"].timestamp()
                row = self.predictor.predict(frame, frame_id)
                row = self.analyze_frame(timestamp, row)
                if bug_traj is not None:
                    row = self.add_bug_traj(row, bug_traj, timestamp)
                rows.append(row)
        finally:
            cap.release()

        if not rows:
            print(f"Skipping empty prediction result: {video_path}")
            return None

        pose_df = pd.concat(rows)
        self.add_run_metadata(pose_df, video_path)
        pose_df.to_parquet(cache_path)
        print(f"Saved: {cache_path}")
        return cache_path

    def add_run_metadata(self, pose_df, video_path):
        metadata = {
            "model_path": self.predictor.model_path,
            "model_name": self.predictor.model_name,
            "cam_name": self.cam_name,
            "calib_dir": str(Path(self.calib_dir).expanduser()) if self.calib_dir else None,
            "calibration_video_date": parse_video_date(video_path),
            "screen_start_x": self.screen_start_x,
            "screen_pix_cm": self.screen_pix_cm,
            "screen_y": self.screen_y,
        }
        pose_df.attrs["run_model_metadata"] = metadata
        for key, value in metadata.items():
            pose_df[(f"run_{key}", "")] = value

    def analyze_frame(self, timestamp, row):
        row[("time", "")] = timestamp
        for col in self.kinematic_cols:
            row[(col, "")] = np.nan

        for bodypart in self.predictor.bodyparts:
            cam_x = row[(bodypart, "cam_x")].iloc[0]
            cam_y = row[(bodypart, "cam_y")].iloc[0]
            x, y = np.nan, np.nan
            if self.caliber is not None and not np.isnan(cam_x) and not np.isnan(cam_y):
                x, y = self.caliber.get_location(cam_x, cam_y)

            if bodypart == self.commit_bodypart:
                if not self.kalman.is_initiated:
                    self.kalman.init(x, y)
                x, y, vx, vy, ax, ay = self.kalman.get_filtered(x, y)
                for col, value in zip(self.kinematic_cols, [x, y, vx, vy, ax, ay]):
                    row.loc[row.index[0], (col, "")] = value
            else:
                row.loc[row.index[0], (bodypart, "x")] = x
                row.loc[row.index[0], (bodypart, "y")] = y

        row.loc[row.index[0], ("angle", "")] = calc_head_angle(row.iloc[0], self.predictor.bodyparts)
        return row

    def add_bug_traj(self, row, bug_traj, timestamp):
        dt = (bug_traj["timestamp"] - timestamp).abs()
        bug_x_cols = [col for col in bug_traj.columns if re.fullmatch(r"bug(?:\d+)?_x", str(col))]
        bug_indices = sorted(
            int(match.group(1)) if (match := re.match(r"bug(\d+)_x", col)) else 0
            for col in bug_x_cols
        )

        for i in bug_indices:
            single = len(bug_indices) == 1 and i == 0
            prefix = "bug" if single else f"bug{i}"
            x_col = f"{prefix}_x"
            y_col = f"{prefix}_y"
            xcm_col = f"{prefix}_x_cm"
            ycm_col = f"{prefix}_y_cm"

            for col in [x_col, y_col, "trial_id", xcm_col, ycm_col]:
                row[(col, "")] = np.nan

            if dt.min() >= 0.03:
                continue

            idx = dt.idxmin()
            for col in [x_col, y_col, "trial_id"]:
                if col in bug_traj.columns:
                    row[(col, "")] = bug_traj.loc[idx, col]

            if self.is_screen_configured and x_col in bug_traj.columns and y_col in bug_traj.columns:
                row[(xcm_col, "")] = self.screen_start_x + bug_traj.loc[idx, x_col] * self.screen_pix_cm
                row[(ycm_col, "")] = self.screen_y + bug_traj.loc[idx, y_col] * self.screen_pix_cm

        return add_gaze_deviation(row, self.screen_y)

    def get_predicted_cache_path(self, video_path):
        preds_dir = Path(video_path).parent / "predictions"
        preds_dir.mkdir(exist_ok=True)
        return preds_dir / f"{self.predictor.model_name}__{Path(video_path).with_suffix('.parquet').name}"


def calc_head_angle(row, bodyparts):
    if not {"nose", "right_ear", "left_ear"}.issubset(set(bodyparts)):
        return np.nan
    x_nose, y_nose = row[("nose", "x")], row[("nose", "y")]
    x_ears = (row[("right_ear", "x")] + row[("left_ear", "x")]) / 2
    y_ears = (row[("right_ear", "y")] + row[("left_ear", "y")]) / 2
    dy = y_ears - y_nose
    dx = x_ears - x_nose
    theta = math.atan(abs(dy) / abs(dx)) if dx != 0 else math.pi / 2
    if dx > 0:
        theta = math.pi - theta
    if dy < 0:
        theta = -theta
    if theta < 0:
        theta = 2 * math.pi + theta
    return theta


def add_gaze_deviation(row, screen_y):
    p = row.iloc[0]
    bug_x_cols = [col for col in row.columns if re.fullmatch(r"bug\d*_x_cm", col[0])]
    for bug_x_col in bug_x_cols:
        match = re.search(r"bug(\d*)_x_cm", bug_x_col[0])
        i = int(match.group(1)) if match and match.group(1).isdigit() else 0
        dev_col = ("dev_angle", "") if i == 0 and len(bug_x_cols) == 1 else (f"dev_angle{i}", "")

        good = (
            p.get(("nose", "prob"), 0) >= 0.8
            and p.get(("left_ear", "prob"), 0) >= 0.8
            and p.get(("right_ear", "prob"), 0) >= 0.8
            and not np.isnan(p[bug_x_col])
            and not np.isnan(p[("nose", "x")])
            and not np.isnan(p[("nose", "y")])
            and screen_y is not None
        )
        if good:
            row.loc[row.index, dev_col] = calc_gaze_deviation_angle(
                p[("angle", "")],
                p[bug_x_col],
                p[("nose", "x")],
                p[("nose", "y")],
                screen_y,
            )
        else:
            row.loc[row.index, dev_col] = np.nan
    return row


def calc_gaze_deviation_angle(angle, bug_x, x, y, screen_y):
    m_exp = (y - screen_y) / (x - bug_x)
    if angle == math.pi / 2:
        x_obs = x
        dev_ang = math.degrees(calc_angle_between_lines(1, 0, m_exp, -1))
    else:
        m_obs = math.tan(math.pi - angle)
        n_obs = y - m_obs * x
        x_obs = (screen_y - n_obs) / m_obs
        a = math.dist((bug_x, screen_y), (x, y))
        b = math.dist((x_obs, screen_y), (x, y))
        c = abs(x_obs - bug_x)
        dev_ang = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
        if angle > math.pi:
            dev_ang = math.pi - dev_ang
        dev_ang = math.degrees(dev_ang)

    sign = np.sign(x_obs - bug_x) if angle < math.pi else -np.sign(x_obs - bug_x)
    return sign * dev_ang


def calc_angle_between_lines(a1, b1, a2, b2):
    return math.acos((a1 * a2 + b1 * b2) / (math.sqrt(a1**2 + b1**2) * math.sqrt(a2**2 + b2**2)))


def load_frames_times(video_path):
    csv_path = Path(video_path).parent / FRAMES_TIMESTAMPS_DIR / Path(video_path).with_suffix(".csv").name
    if not csv_path.exists():
        raise MissingFile(f"Missing frames timestamps: {csv_path}")
    frames_ts = pd.read_csv(csv_path, names=["frame_id", "time"], header=0).set_index("frame_id")
    frames_ts["time"] = pd.to_datetime(frames_ts["time"], unit="s", utc=True).dt.tz_convert("Asia/Jerusalem").dt.tz_localize(None)
    frames_ts.index = frames_ts.index.astype(int)
    return frames_ts


def load_bug_trajectory(video_path):
    csv_path = Path(video_path).parent.parent / "bug_trajectory.csv"
    if not csv_path.exists():
        return None
    bug_traj = pd.read_csv(csv_path, index_col=0)
    bug_traj = rename_bug_columns(bug_traj)
    bug_traj["datetime"] = bug_traj["time"].map(pd.to_datetime).dt.tz_localize(None)
    bug_traj["timestamp"] = bug_traj["datetime"].astype(int).div(10**9)
    return bug_traj.sort_values("datetime").reset_index(drop=True)


def rename_bug_columns(df):
    rename_map = {}
    for col in df.columns:
        match = re.fullmatch(r"([xy])(\d+)$", str(col))
        if match:
            axis, idx = match.groups()
            rename_map[col] = f"bug{idx}_{axis}"
        elif col in {"x", "y"}:
            rename_map[col] = f"bug_{col}"
    return df.rename(columns=rename_map)


def parse_video_date(video_path):
    match = re.match(r"\w+_(\d{8}T\d{6})", Path(video_path).stem)
    return match.group(1) if match else None


def scan_videos(path, cam_name, suffixes):
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    videos = []
    for suffix in suffixes:
        videos.extend(path.rglob(f"{cam_name}*{suffix}"))
    return sorted(v for v in videos if v.is_file() and "predictions" not in v.parts)


def load_video_list(path, cam_name, suffixes):
    videos = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        video_path = Path(line).expanduser()
        if video_path.name.startswith(cam_name) and video_path.suffix.lower() in suffixes:
            videos.append(video_path)
    return videos


def parse_suffixes(text):
    return [suffix if suffix.startswith(".") else f".{suffix}" for suffix in [x.strip().lower() for x in text.split(",")] if suffix]


def main():
    parser = argparse.ArgumentParser(description="Standalone DLC pose prediction without Arena imports.")
    parser.add_argument("-p", "--model_path", required=True, help="Path to exported DLCLive model")
    parser.add_argument("-v", "--video_path", help="Video file or directory to scan")
    parser.add_argument("--video_list_file", help="Newline-separated video paths")
    parser.add_argument("-c", "--cam_name", default="top")
    parser.add_argument("--bodyparts", help="Comma-separated bodyparts if they cannot be read from model YAML")
    parser.add_argument("--model_name", help="Prediction cache prefix; defaults to model folder name")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calib_dir", help="Calibration directory. Omit to skip calibration.")
    parser.add_argument("--video_suffixes", default=".mp4,.avi")
    parser.add_argument("--start_x", type=float)
    parser.add_argument("--pix_cm", type=float)
    parser.add_argument("--screen_y", type=float)
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not args.video_path and not args.video_list_file:
        print("Provide --video_path or --video_list_file", file=sys.stderr)
        sys.exit(1)

    suffixes = parse_suffixes(args.video_suffixes)
    if args.video_list_file:
        videos = load_video_list(args.video_list_file, args.cam_name, set(suffixes))
    else:
        videos = scan_videos(args.video_path, args.cam_name, suffixes)

    if not videos:
        print("No videos found.", file=sys.stderr)
        sys.exit(1)

    bodyparts = load_bodyparts(model_path, args.bodyparts)
    predictor = DLCPredictor(model_path, bodyparts, model_name=args.model_name, threshold=args.threshold)
    runner = PoseRunner(
        predictor,
        args.cam_name,
        calib_dir=args.calib_dir,
        screen_start_x=args.start_x,
        screen_pix_cm=args.pix_cm,
        screen_y=args.screen_y,
    )

    print(f"Model: {model_path}")
    print(f"Cache prefix: {predictor.model_name}")
    print(f"Bodyparts: {', '.join(predictor.bodyparts)}")
    print(f"Calibration: {args.calib_dir or 'off'}")
    print(f"Videos: {len(videos)}")

    saved_or_existing = 0
    skipped_or_failed = 0
    for video in videos:
        print(f"Inferring video: {video}", flush=True)
        runner.is_initialized = False
        try:
            result = runner.predict_video(video, skip_existing=args.skip_existing)
            if result is None:
                skipped_or_failed += 1
            else:
                saved_or_existing += 1
        except Exception as exc:
            skipped_or_failed += 1
            print(f"Skipping failed video: {video}")
            print(f"  {type(exc).__name__}: {exc}")

    print(f"Done. Saved/existing: {saved_or_existing}; skipped/failed: {skipped_or_failed}")


if __name__ == "__main__":
    main()


# Examples:
#   python run_model.py \
#       -p /path/to/exported_dlc_model \
#       -v /path/to/block/videos \
#       -c top \
#       --bodyparts nose,left_ear,right_ear
#
#   python run_model.py \
#       -p /path/to/exported_dlc_model \
#       -v /path/to/block/videos \
#       -c top \
#       --calib_dir /path/to/calibrations \
#       --start_x 7.59 --pix_cm 0.027604 --screen_y -4.3
#
# Output is saved under:
#   /path/to/block/videos/predictions/<model_name>__<video_name>.parquet



# videos="/media/sil3/Data/Bareket/experiments/reptilearn5/PV252"
# calib="/media/sil3/Data/Bareket/arena_configs/reptilearn5/calibrations"
# model="/media/sil4/Data1/Bareket/models/deeplabcut/front_top_head_resnet_152_retrained"
# cam="top"
# python run_model.py -p $model -v $videos -c $cam --calib_dir $calib --skip_existing --start_x 7.59 --pix_cm 0.027604 --screen_y -4.3
