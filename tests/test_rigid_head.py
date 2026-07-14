import numpy as np
import pandas as pd

from rigid_head import RIGID_COLUMNS, add_rigid_head_correction, head_overlay_columns
import run_edited_pose_calibration as calibration


LANDMARKS = ("nose", "left_ear", "right_ear")
CANONICAL = np.array([[5.0, 0.0], [0.0, 2.0], [0.0, -2.0]])


def transformed_triangle(frame, scale=1.0):
    angle = frame * 0.05
    rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    return scale * CANONICAL @ rotation + np.array([100 + frame, 200 + frame * 0.5])


def one_trial():
    rows = []
    for frame in range(21):
        points = transformed_triangle(frame)
        row = {"trial_id": 8875, "frame_idx": frame}
        for index, name in enumerate(LANDMARKS):
            row[f"{name}_cam_x"] = points[index, 0]
            row[f"{name}_cam_y"] = points[index, 1]
            row[f"{name}_prob"] = 1.0
            row[f"{name}_x"] = 1000 + frame
            row[f"{name}_y"] = 2000 + frame
        rows.append(row)

    return pd.DataFrame(rows, index=np.arange(100, 121))


def rigid_side_lengths(row):
    points = np.array([
        [row[f"rigid_{name}_cam_x"], row[f"rigid_{name}_cam_y"]]
        for name in LANDMARKS
    ])
    return np.array([
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[1] - points[2]),
    ])


def test_rigid_fit_uses_camera_columns_and_never_deforms_triangle():
    raw = one_trial()
    raw.loc[110, ["nose_cam_x", "nose_cam_y", "nose_prob"]] = [np.nan, np.nan, 0.0]
    raw.loc[111, ["right_ear_cam_x", "right_ear_cam_y", "right_ear_prob"]] = [np.nan, np.nan, 0.0]
    raw.loc[112, ["left_ear_cam_x", "left_ear_cam_y", "left_ear_prob"]] = [np.nan, np.nan, 0.0]
    corrected = add_rigid_head_correction(raw)

    assert corrected.index.equals(raw.index)
    pd.testing.assert_frame_equal(corrected[raw.columns], raw)
    assert corrected.loc[100, "rigid_method"] == "raw_good_triangle"
    assert not corrected.loc[100, "is_rigid"]
    assert (corrected.loc[110:112, "rigid_method"] == "kalman_missing").all()
    assert corrected.loc[110:112, "is_rigid"].all()
    assert corrected.loc[110, "rigid_changed_points"] == "nose"
    assert corrected.loc[111, "rigid_changed_points"] == "right_ear"
    assert corrected.loc[112, "rigid_changed_points"] == "left_ear"

    lengths = np.vstack([
        rigid_side_lengths(corrected.loc[index]) for index in range(109, 114)
    ])
    np.testing.assert_allclose(lengths, np.tile(lengths[0], (len(lengths), 1)), atol=1e-10)
    assert corrected.loc[100, "rigid_nose_cam_x"] < 500


def test_high_confidence_unlikely_triangle_is_corrected():
    raw = one_trial()
    raw.loc[110, "nose_cam_x"] += 50

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[109, "rigid_method"] == "raw_good_triangle"
    assert corrected.loc[110, "rigid_method"] == "kalman_geometry"
    assert corrected.loc[110, "is_rigid"]
    assert corrected.loc[110, "rigid_changed_points"] == "nose"
    np.testing.assert_allclose(
        rigid_side_lengths(corrected.loc[110]),
        rigid_side_lengths(corrected.loc[109]),
        rtol=0.05,
    )


def test_low_probability_alone_triggers_rigid_replacement():
    raw = one_trial()
    raw.loc[110, "right_ear_prob"] = 0.2

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[110, "rigid_method"] == "kalman_low_confidence"
    assert corrected.loc[110, "rigid_changed_points"] == "right_ear"
    assert corrected.loc[110, "is_rigid"]
    for name in ("nose", "left_ear"):
        np.testing.assert_allclose(
            corrected.loc[110, [f"rigid_{name}_cam_x", f"rigid_{name}_cam_y"]].astype(float),
            raw.loc[110, [f"{name}_cam_x", f"{name}_cam_y"]].astype(float),
        )


def test_low_probability_replacement_uses_the_local_shape():
    raw = one_trial()
    for offset, index in enumerate(raw.index):
        raw.loc[index, "nose_cam_x"] += offset * 0.15
    raw.loc[110, "right_ear_prob"] = 0.2

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[110, "is_rigid"]
    np.testing.assert_allclose(
        corrected.loc[110, ["rigid_right_ear_cam_x", "rigid_right_ear_cam_y"]].astype(float),
        raw.loc[110, ["right_ear_cam_x", "right_ear_cam_y"]].astype(float),
        atol=0.5,
    )


def test_one_reliable_landmark_anchors_the_kalman_pose():
    raw = one_trial()
    raw.loc[110, ["left_ear_cam_x", "left_ear_cam_y", "left_ear_prob"]] = [150, 250, 0.2]
    raw.loc[110, ["right_ear_cam_x", "right_ear_cam_y", "right_ear_prob"]] = [50, 150, 0.2]

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[110, "rigid_method"] == "kalman_low_confidence"
    assert corrected.loc[110, "rigid_changed_points"] == "left_ear, right_ear"
    assert corrected.loc[110, "rigid_n_observations"] == 1
    np.testing.assert_allclose(
        corrected.loc[110, ["rigid_nose_cam_x", "rigid_nose_cam_y"]].astype(float),
        raw.loc[110, ["nose_cam_x", "nose_cam_y"]].astype(float),
    )
    np.testing.assert_allclose(
        rigid_side_lengths(corrected.loc[110]),
        rigid_side_lengths(corrected.loc[109]),
        rtol=0.1,
    )


def test_kalman_rigid_pose_replaces_a_deviant_low_confidence_point():
    raw = one_trial()
    raw.loc[110, "right_ear_cam_x"] += 20
    raw.loc[110, "right_ear_prob"] = 0.2

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[110, "rigid_method"] == "kalman_low_confidence"
    assert corrected.loc[110, "rigid_changed_points"] == "right_ear"
    assert corrected.loc[110, "is_rigid"]
    for name in ("nose", "left_ear"):
        np.testing.assert_allclose(
            corrected.loc[110, [f"rigid_{name}_cam_x", f"rigid_{name}_cam_y"]].astype(float),
            raw.loc[110, [f"{name}_cam_x", f"{name}_cam_y"]].astype(float),
        )


def test_kalman_does_not_guess_a_fully_missing_gap_without_current_anchors():
    raw = one_trial()
    for name in LANDMARKS:
        raw.loc[110, [f"{name}_cam_x", f"{name}_cam_y", f"{name}_prob"]] = [
            np.nan, np.nan, 0.0
        ]

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[110, "rigid_method"] == "kalman_unresolved"
    assert not corrected.loc[110, "is_rigid"]


def test_problematic_frame_without_nearby_pose_is_marked_unresolved():
    raw = one_trial()
    for index in range(100, 111):
        for name in LANDMARKS:
            raw.loc[index, f"{name}_prob"] = 0.0
    raw.loc[100, "nose_cam_x"] += 50

    corrected = add_rigid_head_correction(raw)

    assert corrected.loc[100, "rigid_method"] == "kalman_unresolved"
    assert not corrected.loc[100, "is_rigid"]


def test_similarity_fit_keeps_two_trusted_anchors_and_recalculates_midpoint():
    raw = one_trial()
    points = transformed_triangle(10, scale=1.5)
    for point_index, name in enumerate(LANDMARKS):
        raw.loc[110, f"{name}_cam_x"] = points[point_index, 0]
        raw.loc[110, f"{name}_cam_y"] = points[point_index, 1]
    raw.loc[110, ["nose_cam_x", "nose_cam_y", "nose_prob"]] = [np.nan, np.nan, 0.0]

    corrected = add_rigid_head_correction(raw)

    for name in ("left_ear", "right_ear"):
        np.testing.assert_allclose(
            corrected.loc[110, [f"rigid_{name}_cam_x", f"rigid_{name}_cam_y"]].astype(float),
            raw.loc[110, [f"{name}_cam_x", f"{name}_cam_y"]].astype(float),
        )
    expected_midpoint = corrected.loc[110, [
        "rigid_left_ear_cam_x", "rigid_left_ear_cam_y"
    ]].to_numpy(dtype=float) / 2 + corrected.loc[110, [
        "rigid_right_ear_cam_x", "rigid_right_ear_cam_y"
    ]].to_numpy(dtype=float) / 2
    np.testing.assert_allclose(
        corrected.loc[110, ["rigid_mid_ears_cam_x", "rigid_mid_ears_cam_y"]].astype(float),
        expected_midpoint,
    )


def test_apply_twice_replaces_columns_without_changing_rows():
    once = add_rigid_head_correction(one_trial())
    twice = add_rigid_head_correction(once)

    assert len(twice) == len(once)
    assert twice.index.equals(once.index)
    assert all(list(twice.columns).count(column) == 1 for column in RIGID_COLUMNS)
    pd.testing.assert_frame_equal(twice[RIGID_COLUMNS], once[RIGID_COLUMNS])


def test_overlay_and_calibration_use_rigid_camera_columns():
    corrected = add_rigid_head_correction(one_trial())
    raw_mapping, raw_available = head_overlay_columns("Raw", corrected.columns)
    rigid_mapping, rigid_available = head_overlay_columns("Rigid", corrected.columns)

    assert raw_available and rigid_available
    assert raw_mapping["nose"] == ("nose_cam_x", "nose_cam_y")
    assert rigid_mapping["nose"] == ("rigid_nose_cam_x", "rigid_nose_cam_y")
    bodyparts = calibration.detect_bodyparts(corrected)
    assert "nose" in bodyparts
    assert "rigid_nose" in bodyparts


def test_reapply_calibration_creates_rigid_world_columns(tmp_path, monkeypatch):
    class FakeCalibration:
        def __init__(self, calib_dir, cam_name):
            pass

        def init(self, image_date):
            return tmp_path / "undistortion.pkl", tmp_path / "transform.pkl"

        def get_location(self, x, y):
            return x / 10.0, y / 10.0

    monkeypatch.setattr(calibration, "CharucoCalibration", FakeCalibration)
    pose_path = tmp_path / "prediction.csv"
    corrected = add_rigid_head_correction(one_trial()).iloc[:5]
    corrected.to_csv(pose_path, index=False)

    calibration.run_edited_pose_calibration_job(
        str(pose_path), "top_20260107T120100.mp4", str(tmp_path)
    )
    calibrated = pd.read_csv(pose_path)

    np.testing.assert_allclose(
        calibrated["rigid_nose_x"], calibrated["rigid_nose_cam_x"] / 10.0
    )
    available = calibrated["rigid_nose_cam_x"].notna()
    assert calibrated.loc[available, "rigid_head_angle_rad"].notna().all()
