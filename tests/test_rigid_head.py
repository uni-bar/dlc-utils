import numpy as np
import pandas as pd

from rigid_head import RIGID_COLUMNS, RigidHeadConfig, add_rigid_head_correction, head_overlay_columns


def one_trial():
    rows = []
    for frame in range(10):
        rows.append({
            "trial_id": 8875,
            "frame_idx": frame,
            "nose_x": 5.0 + frame * 0.1,
            "nose_y": 0.0,
            "nose_prob": 1.0,
            "left_ear_x": frame * 0.1,
            "left_ear_y": 2.0,
            "left_ear_prob": 1.0,
            "right_ear_x": frame * 0.1,
            "right_ear_y": -2.0,
            "right_ear_prob": 1.0,
        })
    df = pd.DataFrame(rows, index=np.arange(100, 110))
    df.loc[101, ["nose_x", "nose_y", "nose_prob"]] = [np.nan, np.nan, 0.0]
    df.loc[102, ["right_ear_x", "right_ear_y", "right_ear_prob"]] = [np.nan, np.nan, 0.0]
    df.loc[103, ["left_ear_x", "left_ear_y", "left_ear_prob"]] = [np.nan, np.nan, 0.0]
    df.loc[104, ["left_ear_x", "left_ear_y", "left_ear_prob",
                 "right_ear_x", "right_ear_y", "right_ear_prob"]] = [np.nan] * 4 + [0.0, 0.0]
    for index in range(105, 110):
        for name in ("nose", "left_ear", "right_ear"):
            df.loc[index, [f"{name}_x", f"{name}_y", f"{name}_prob"]] = [np.nan, np.nan, 0.0]
    return df


def test_methods_flags_and_continuity():
    raw = one_trial()
    corrected = add_rigid_head_correction(raw, RigidHeadConfig(max_carry_gap=3))

    assert corrected.index.equals(raw.index)
    pd.testing.assert_frame_equal(corrected[raw.columns], raw)
    assert corrected["is_rigid"].dtype == bool
    assert corrected.loc[100, "rigid_method"] == "raw_good_triangle"
    assert not corrected.loc[100, "is_rigid"]
    assert corrected.loc[101, "rigid_method"] == "infer_nose_from_ears"
    assert corrected.loc[102, "rigid_method"] == "infer_right_ear"
    assert corrected.loc[103, "rigid_method"] == "infer_left_ear"
    assert corrected.loc[104, "rigid_method"] == "infer_both_ears"
    assert list(corrected.loc[105:107, "rigid_method"]) == ["carry_previous"] * 3
    assert corrected.loc[108, "rigid_method"] == "unavailable"
    assert not corrected.loc[108, "is_rigid"]
    inferred_angles = corrected.loc[100:104, "rigid_head_angle_rad"].to_numpy()
    wrapped_changes = np.arctan2(np.sin(np.diff(inferred_angles)), np.cos(np.diff(inferred_angles)))
    assert np.max(np.abs(wrapped_changes)) < np.pi / 2


def test_apply_twice_replaces_columns_without_changing_rows():
    once = add_rigid_head_correction(one_trial())
    twice = add_rigid_head_correction(once)

    assert len(twice) == len(once)
    assert twice.index.equals(once.index)
    assert all(list(twice.columns).count(column) == 1 for column in RIGID_COLUMNS)
    pd.testing.assert_frame_equal(twice[RIGID_COLUMNS], once[RIGID_COLUMNS])


def test_overlay_mode_selects_columns_without_recomputation():
    corrected = add_rigid_head_correction(one_trial())
    raw_mapping, raw_available = head_overlay_columns("Raw", corrected.columns)
    rigid_mapping, rigid_available = head_overlay_columns("Rigid", corrected.columns)

    assert raw_available and rigid_available
    assert raw_mapping["nose"] == ("nose_x", "nose_y")
    assert rigid_mapping["nose"] == ("rigid_nose_x", "rigid_nose_y")
