"""Simple geometry-based rigid head reconstruction for DLC trajectories."""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


LANDMARKS = ("nose", "left_ear", "right_ear")
RIGID_COLUMNS = [
    "rigid_nose_x", "rigid_nose_y",
    "rigid_left_ear_x", "rigid_left_ear_y",
    "rigid_right_ear_x", "rigid_right_ear_y",
    "rigid_mid_ears_x", "rigid_mid_ears_y",
    "rigid_head_angle_rad", "rigid_head_angle_deg",
    "rigid_method", "rigid_cost", "rigid_n_candidates", "is_rigid",
]


@dataclass
class RigidHeadConfig:
    confidence_threshold: float = 0.8
    min_length: float = 1e-6
    geometry_ratio_min: float = 0.55
    geometry_ratio_max: float = 1.8
    max_carry_gap: int = 3
    angle_weight: float = 2.0
    midpoint_weight: float = 1.0
    landmark_weight: float = 0.5
    observation_weight: float = 2.0


def head_overlay_columns(mode: str, columns: Iterable[str]) -> Tuple[Dict[str, Tuple[str, str]], bool]:
    """Return the columns used by the raw/rigid head overlay."""
    columns = set(columns)
    raw = {name: (f"{name}_x", f"{name}_y") for name in LANDMARKS}
    rigid = {name: (f"rigid_{name}_x", f"rigid_{name}_y") for name in LANDMARKS}
    mapping = rigid if mode.lower() == "rigid" else raw
    available = all(x in columns and y in columns for x, y in mapping.values())
    return mapping, available


def _point(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"{name}_x"], row[f"{name}_y"]], dtype=float)


def _confidence_column(df: pd.DataFrame, name: str) -> str:
    for suffix in ("prob", "likelihood", "conf"):
        column = f"{name}_{suffix}"
        if column in df.columns:
            return column
    raise ValueError(f"Missing confidence column for {name}")


def _finite(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _shape(triangle: Dict[str, np.ndarray]) -> Tuple[float, float]:
    midpoint = (triangle["left_ear"] + triangle["right_ear"]) / 2.0
    return (
        float(np.linalg.norm(triangle["nose"] - midpoint)),
        float(np.linalg.norm(triangle["left_ear"] - triangle["right_ear"]) / 2.0),
    )


def _valid_shape(triangle: Dict[str, np.ndarray], head_length: float, half_width: float,
                 config: RigidHeadConfig) -> bool:
    if not all(_finite(point) for point in triangle.values()):
        return False
    length, width = _shape(triangle)
    if length <= config.min_length or width <= config.min_length:
        return False
    return (
        config.geometry_ratio_min <= length / head_length <= config.geometry_ratio_max
        and config.geometry_ratio_min <= width / half_width <= config.geometry_ratio_max
    )


def _angle(triangle: Dict[str, np.ndarray]) -> float:
    midpoint = (triangle["left_ear"] + triangle["right_ear"]) / 2.0
    direction = triangle["nose"] - midpoint
    return float(np.arctan2(direction[1], direction[0]))


def _angle_difference(a: float, b: float) -> float:
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def _triangle_from_pose(nose: np.ndarray, direction: np.ndarray, head_length: float,
                        half_width: float) -> Dict[str, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    midpoint = nose - head_length * direction
    perpendicular = np.array([-direction[1], direction[0]])
    return {
        "nose": nose.copy(),
        "left_ear": midpoint + half_width * perpendicular,
        "right_ear": midpoint - half_width * perpendicular,
    }


def _candidate_cost(candidate: Dict[str, np.ndarray], previous: Dict[str, np.ndarray],
                    observed: Dict[str, np.ndarray], reliable: Dict[str, bool],
                    scale: float, config: RigidHeadConfig) -> float:
    if previous is None:
        temporal = 0.0
    else:
        midpoint = (candidate["left_ear"] + candidate["right_ear"]) / 2.0
        previous_midpoint = (previous["left_ear"] + previous["right_ear"]) / 2.0
        angle_cost = _angle_difference(_angle(candidate), _angle(previous))
        midpoint_cost = np.linalg.norm(midpoint - previous_midpoint) / scale
        landmark_cost = np.mean([
            np.linalg.norm(candidate[name] - previous[name]) / scale for name in LANDMARKS
        ])
        temporal = (
            config.angle_weight * angle_cost
            + config.midpoint_weight * midpoint_cost
            + config.landmark_weight * landmark_cost
        )
    observation = sum(
        np.linalg.norm(candidate[name] - observed[name]) / scale
        for name in LANDMARKS if reliable[name]
    )
    return float(temporal + config.observation_weight * observation)


def _canonical_shape(group: pd.DataFrame, confidence_columns: Dict[str, str],
                     config: RigidHeadConfig) -> Tuple[float, float]:
    lengths = []
    widths = []
    for _, row in group.iterrows():
        points = {name: _point(row, name) for name in LANDMARKS}
        reliable = all(
            _finite(points[name]) and float(row[confidence_columns[name]]) >= config.confidence_threshold
            for name in LANDMARKS
        )
        if reliable:
            length, width = _shape(points)
            if length > config.min_length and width > config.min_length:
                lengths.append(length)
                widths.append(width)
    if not lengths:
        return np.nan, np.nan
    return float(np.median(lengths)), float(np.median(widths))


def _candidates(observed: Dict[str, np.ndarray], reliable: Dict[str, bool],
                previous: Dict[str, np.ndarray], head_length: float,
                half_width: float):
    nose_ok, left_ok, right_ok = (reliable[name] for name in LANDMARKS)
    if left_ok and right_ok and not nose_ok:
        midpoint = (observed["left_ear"] + observed["right_ear"]) / 2.0
        ear_axis = observed["left_ear"] - observed["right_ear"]
        perpendicular = np.array([-ear_axis[1], ear_axis[0]]) / np.linalg.norm(ear_axis)
        return [
            ({"nose": midpoint + sign * head_length * perpendicular,
              "left_ear": observed["left_ear"].copy(),
              "right_ear": observed["right_ear"].copy()}, "infer_nose_from_ears")
            for sign in (1.0, -1.0)
        ]
    if nose_ok and left_ok and not right_ok:
        vector = observed["left_ear"] - observed["nose"]
        theta = np.arctan2(vector[1], vector[0]) - np.arctan2(half_width, -head_length)
        direction = np.array([np.cos(theta), np.sin(theta)])
        triangle = _triangle_from_pose(observed["nose"], direction, head_length, half_width)
        triangle["left_ear"] = observed["left_ear"].copy()
        return [(triangle, "infer_right_ear")]
    if nose_ok and right_ok and not left_ok:
        vector = observed["right_ear"] - observed["nose"]
        theta = np.arctan2(vector[1], vector[0]) - np.arctan2(-half_width, -head_length)
        direction = np.array([np.cos(theta), np.sin(theta)])
        triangle = _triangle_from_pose(observed["nose"], direction, head_length, half_width)
        triangle["right_ear"] = observed["right_ear"].copy()
        return [(triangle, "infer_left_ear")]
    if nose_ok and not left_ok and not right_ok and previous is not None:
        direction = previous["nose"] - (previous["left_ear"] + previous["right_ear"]) / 2.0
        return [(_triangle_from_pose(observed["nose"], direction, head_length, half_width), "infer_both_ears")]
    return []


def add_rigid_head_correction(trajectories_df: pd.DataFrame,
                              config: RigidHeadConfig = None) -> pd.DataFrame:
    """Add rigid head coordinates without changing the raw landmark columns."""
    config = config or RigidHeadConfig()
    required = [f"{name}_{axis}" for name in LANDMARKS for axis in ("x", "y")]
    missing = [column for column in required if column not in trajectories_df.columns]
    if missing:
        raise ValueError(f"Missing head coordinate columns: {', '.join(missing)}")
    confidence_columns = {
        name: _confidence_column(trajectories_df, name) for name in LANDMARKS
    }
    result = trajectories_df.drop(columns=RIGID_COLUMNS, errors="ignore").copy()
    output = pd.DataFrame(index=result.index)
    for column in RIGID_COLUMNS:
        output[column] = False if column == "is_rigid" else np.nan
    output["rigid_method"] = "unavailable"
    output["rigid_n_candidates"] = 0

    trial_column = "trial_id" if "trial_id" in result.columns else "trial_id_" if "trial_id_" in result.columns else None
    groups = result.groupby(trial_column, sort=False, dropna=False) if trial_column else [(None, result)]
    for _, group in groups:
        for ordering_column in ("frame_idx", "frame", "time", "timestamp"):
            if ordering_column in group.columns:
                group = group.sort_values(ordering_column, kind="stable")
                break
        head_length, half_width = _canonical_shape(group, confidence_columns, config)
        if not np.isfinite(head_length) or not np.isfinite(half_width):
            continue
        previous = None
        carry_gap = 0
        for index, row in group.iterrows():
            observed = {name: _point(row, name) for name in LANDMARKS}
            reliable = {
                name: _finite(observed[name])
                and pd.notna(row[confidence_columns[name]])
                and float(row[confidence_columns[name]]) >= config.confidence_threshold
                for name in LANDMARKS
            }
            raw_good = all(reliable.values()) and _valid_shape(
                observed, head_length, half_width, config
            )
            if raw_good:
                candidates = [(observed, "raw_good_triangle")]
            else:
                candidates = _candidates(
                    observed, reliable, previous, head_length, half_width
                )
                candidates = [
                    (triangle, method) for triangle, method in candidates
                    if _valid_shape(triangle, head_length, half_width, config)
                ]

            if candidates:
                costs = [
                    _candidate_cost(triangle, previous, observed, reliable, head_length, config)
                    for triangle, _ in candidates
                ]
                selected = int(np.argmin(costs))
                triangle, method = candidates[selected]
                cost = costs[selected]
                carry_gap = 0
            elif previous is not None and carry_gap < config.max_carry_gap:
                triangle = {name: point.copy() for name, point in previous.items()}
                method = "carry_previous"
                cost = _candidate_cost(triangle, previous, observed, reliable, head_length, config)
                carry_gap += 1
                candidates = [(triangle, method)]
            else:
                previous = None
                carry_gap = 0
                continue

            midpoint = (triangle["left_ear"] + triangle["right_ear"]) / 2.0
            angle = _angle(triangle)
            for name in LANDMARKS:
                output.at[index, f"rigid_{name}_x"] = triangle[name][0]
                output.at[index, f"rigid_{name}_y"] = triangle[name][1]
            output.at[index, "rigid_mid_ears_x"] = midpoint[0]
            output.at[index, "rigid_mid_ears_y"] = midpoint[1]
            output.at[index, "rigid_head_angle_rad"] = angle
            output.at[index, "rigid_head_angle_deg"] = np.degrees(angle)
            output.at[index, "rigid_method"] = method
            output.at[index, "rigid_cost"] = cost
            output.at[index, "rigid_n_candidates"] = len(candidates)
            output.at[index, "is_rigid"] = method not in ("raw_good_triangle", "unavailable")
            previous = triangle

    output["is_rigid"] = output["is_rigid"].astype(bool)
    output["rigid_n_candidates"] = output["rigid_n_candidates"].astype(int)
    return result.join(output)


def rigid_head_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return the concise diagnostics shown by the UI."""
    method = df["rigid_method"]
    corrected = df["is_rigid"].astype(bool)
    trial_column = "trial_id" if "trial_id" in df.columns else "trial_id_" if "trial_id_" in df.columns else None
    return {
        "total_rows": len(df),
        "raw_valid_rows": int((method == "raw_good_triangle").sum()),
        "corrected_rows": int(corrected.sum()),
        "carried_rows": int((method == "carry_previous").sum()),
        "unavailable_rows": int((method == "unavailable").sum()),
        "corrected_trials": int(df.loc[corrected, trial_column].nunique()) if trial_column else int(corrected.any()),
    }
