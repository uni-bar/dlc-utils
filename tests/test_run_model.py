import pandas as pd

from run_model import load_bug_trajectory


def test_bug_trajectory_accepts_mixed_iso_timestamp_precision(tmp_path):
    video_path = tmp_path / "block" / "videos" / "top_20260107T120100.mp4"
    video_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "time": [
            "2026-01-07T12:03:06.123456+02:00",
            "2026-01-07T12:03:07+02:00",
        ],
        "x": [1, 2],
        "y": [3, 4],
    }).to_csv(video_path.parent.parent / "bug_trajectory.csv")

    trajectory = load_bug_trajectory(video_path)

    assert len(trajectory) == 2
    assert trajectory["timestamp"].notna().all()
