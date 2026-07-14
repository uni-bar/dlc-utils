"""Conservative rigid reconstruction of nose and ears in camera coordinates."""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


LANDMARKS = ("nose", "left_ear", "right_ear")
RIGID_CAMERA_COLUMNS = [
    f"rigid_{name}_cam_{axis}"
    for name in (*LANDMARKS, "mid_ears")
    for axis in ("x", "y")
]
RIGID_DIAGNOSTIC_COLUMNS = [
    "rigid_head_cam_angle_rad",
    "rigid_head_cam_angle_deg",
    "rigid_method",
    "rigid_fit_error",
    "rigid_n_observations",
    "rigid_changed_points",
    "is_rigid",
]
RIGID_COLUMNS = RIGID_CAMERA_COLUMNS + RIGID_DIAGNOSTIC_COLUMNS
STALE_RIGID_COLUMNS = RIGID_COLUMNS + [
    f"rigid_{name}_{axis}"
    for name in (*LANDMARKS, "mid_ears")
    for axis in ("x", "y")
] + [
    "rigid_head_angle_rad",
    "rigid_head_angle_deg",
    "rigid_cost",
    "rigid_n_candidates",
]


@dataclass
class RigidHeadConfig:
    confidence_threshold: float = 0.8
    correction_confidence_threshold: float = 0.5
    window_radius: int = 10
    seed_shape_error: float = 0.5
    suspicious_shape_error: float = 0.15
    point_outlier_error: float = 0.5


def head_overlay_columns(mode: str, columns: Iterable[str]) -> Tuple[Dict[str, Tuple[str, str]], bool]:
    """Return camera-coordinate columns for the raw or rigid video overlay."""
    columns = set(columns)
    prefix = "rigid_" if mode.lower() == "rigid" else ""
    mapping = {
        name: (f"{prefix}{name}_cam_x", f"{prefix}{name}_cam_y")
        for name in LANDMARKS
    }
    available = all(x in columns and y in columns for x, y in mapping.values())
    return mapping, available


def _confidence_column(df: pd.DataFrame, name: str) -> str:
    for suffix in ("prob", "likelihood", "conf"):
        column = f"{name}_{suffix}"
        if column in df.columns:
            return column
    raise ValueError(f"Missing confidence column for {name}")


def _points(row: pd.Series) -> np.ndarray:
    return np.array([
        [row[f"{name}_cam_x"], row[f"{name}_cam_y"]]
        for name in LANDMARKS
    ], dtype=float)


def _side_lengths(points: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[1] - points[2]),
    ])


def _shape_signature(points: np.ndarray) -> np.ndarray:
    sides = _side_lengths(points)
    mean_side = sides.mean()
    if mean_side == 0:
        raise ValueError("Degenerate head triangle")
    return sides / mean_side


def _normalized_triangle(points: np.ndarray) -> np.ndarray:
    ear_midpoint = (points[1] + points[2]) / 2.0
    direction = points[0] - ear_midpoint
    length = np.linalg.norm(direction)
    if length == 0 or np.linalg.norm(points[1] - points[2]) == 0:
        raise ValueError("Degenerate head triangle")
    direction /= length
    perpendicular = np.array([-direction[1], direction[0]])
    return (points - ear_midpoint) @ np.column_stack([direction, perpendicular])


def _fit_similarity(template: np.ndarray, observed: np.ndarray,
                    anchor_indices: np.ndarray) -> np.ndarray:
    """Fit translation, rotation, and camera-space scale to trusted anchors."""
    source = template[anchor_indices]
    target = observed[anchor_indices]
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, singular_values, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = singular_values.sum() / np.sum(centered_source ** 2)
    translation = target_center - scale * source_center @ rotation
    return scale * template @ rotation + translation


def _head_angle(points: np.ndarray) -> float:
    midpoint = (points[1] + points[2]) / 2.0
    direction = points[0] - midpoint
    return float(np.arctan2(direction[1], direction[0]))


def _trusted_seed_mask(points: np.ndarray, confident: np.ndarray,
                       config: RigidHeadConfig) -> np.ndarray:
    """Find complete high-confidence frames with locally typical shape ratios."""
    candidates = confident.all(axis=1)
    signatures = np.full((len(points), 3), np.nan)
    for position in np.flatnonzero(candidates):
        try:
            signatures[position] = _shape_signature(points[position])
        except ValueError:
            candidates[position] = False

    seed = np.zeros(len(points), dtype=bool)
    candidate_positions = np.flatnonzero(candidates)
    if not len(candidate_positions):
        return seed
    global_signature = np.nanmedian(signatures[candidate_positions], axis=0)

    for position in candidate_positions:
        neighbors = candidate_positions[
            (candidate_positions >= position - config.window_radius)
            & (candidate_positions <= position + config.window_radius)
            & (candidate_positions != position)
        ]
        reference = (
            np.nanmedian(signatures[neighbors], axis=0)
            if len(neighbors) >= 2 else global_signature
        )
        error = np.max(np.abs(signatures[position] / reference - 1.0))
        seed[position] = error <= config.seed_shape_error
    return seed


def _pose_from_triangle(template: np.ndarray, points: np.ndarray) -> np.ndarray:
    midpoint = (points[1] + points[2]) / 2.0
    scale = _side_lengths(points).mean() / _side_lengths(template).mean()
    return np.array([midpoint[0], midpoint[1], _head_angle(points), np.log(scale)])


def _triangle_from_pose(template: np.ndarray, pose: np.ndarray) -> np.ndarray:
    angle = pose[2]
    rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    return np.exp(pose[3]) * template @ rotation + pose[:2]


def _kalman_smooth(values: np.ndarray, process_variance: float,
                   measurement_variance: float) -> np.ndarray:
    """Constant-velocity Kalman filter followed by an RTS backward pass."""
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    smoothed = np.full(len(values), np.nan)
    if not len(valid):
        return smoothed

    start, stop = valid[0], valid[-1]
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])
    process_noise = process_variance * np.array([[0.25, 0.5], [0.5, 1.0]])
    measurement_noise = np.array([[measurement_variance]])
    identity = np.eye(2)

    filtered_state = np.full((len(values), 2), np.nan)
    filtered_cov = np.full((len(values), 2, 2), np.nan)
    predicted_state = np.full((len(values), 2), np.nan)
    predicted_cov = np.full((len(values), 2, 2), np.nan)

    state = np.array([values[start], 0.0])
    covariance = np.diag([measurement_variance, process_variance + measurement_variance])
    for position in range(start, stop + 1):
        if position > start:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        predicted_state[position] = state
        predicted_cov[position] = covariance

        if np.isfinite(values[position]):
            residual = values[position] - float(observation @ state)
            residual_cov = observation @ covariance @ observation.T + measurement_noise
            gain = covariance @ observation.T / residual_cov[0, 0]
            state = state + gain[:, 0] * residual
            covariance = (identity - gain @ observation) @ covariance
        filtered_state[position] = state
        filtered_cov[position] = covariance

    smooth_state = filtered_state.copy()
    for position in range(stop - 1, start - 1, -1):
        gain = (
            filtered_cov[position]
            @ transition.T
            @ np.linalg.inv(predicted_cov[position + 1])
        )
        smooth_state[position] += gain @ (
            smooth_state[position + 1] - predicted_state[position + 1]
        )
    smoothed[start:stop + 1] = smooth_state[start:stop + 1, 0]
    return smoothed


def _smooth_pose_measurements(measurements: np.ndarray) -> np.ndarray:
    result = np.full_like(measurements, np.nan)
    result[:, 0] = _kalman_smooth(measurements[:, 0], 1.0, 4.0)
    result[:, 1] = _kalman_smooth(measurements[:, 1], 1.0, 4.0)

    angles = measurements[:, 2].copy()
    valid_angles = np.flatnonzero(np.isfinite(angles))
    if len(valid_angles):
        angles[valid_angles] = np.unwrap(angles[valid_angles])
    result[:, 2] = _kalman_smooth(angles, 0.0025, 0.01)
    result[:, 3] = _kalman_smooth(measurements[:, 3], 0.0004, 0.0025)
    return result


def _write_output(output: Dict[str, np.ndarray], position: int, points: np.ndarray, method: str,
                  fit_error: float, n_observations: int, corrected: bool,
                  changed_points: str = ""):
    midpoint = (points[1] + points[2]) / 2.0
    angle = _head_angle(points)
    for point_index, name in enumerate(LANDMARKS):
        output[f"rigid_{name}_cam_x"][position] = points[point_index, 0]
        output[f"rigid_{name}_cam_y"][position] = points[point_index, 1]
    output["rigid_mid_ears_cam_x"][position] = midpoint[0]
    output["rigid_mid_ears_cam_y"][position] = midpoint[1]
    output["rigid_head_cam_angle_rad"][position] = angle
    output["rigid_head_cam_angle_deg"][position] = np.degrees(angle)
    output["rigid_method"][position] = method
    output["rigid_fit_error"][position] = fit_error
    output["rigid_n_observations"][position] = n_observations
    output["rigid_changed_points"][position] = changed_points
    output["is_rigid"][position] = corrected


def add_rigid_head_correction(trajectories_df: pd.DataFrame,
                              config: RigidHeadConfig = None) -> pd.DataFrame:
    """Correct suspicious frames using trustworthy frames within a ±10 window."""
    config = config or RigidHeadConfig()
    required = [f"{name}_cam_{axis}" for name in LANDMARKS for axis in ("x", "y")]
    missing = [column for column in required if column not in trajectories_df.columns]
    if missing:
        raise ValueError(f"Missing head camera-coordinate columns: {', '.join(missing)}")

    confidence_columns = {
        name: _confidence_column(trajectories_df, name) for name in LANDMARKS
    }
    result = trajectories_df.drop(columns=STALE_RIGID_COLUMNS, errors="ignore").copy()
    output = {}
    for column in RIGID_CAMERA_COLUMNS + [
        "rigid_head_cam_angle_rad", "rigid_head_cam_angle_deg", "rigid_fit_error"
    ]:
        output[column] = np.full(len(result), np.nan)
    output["rigid_method"] = np.full(len(result), "unavailable", dtype=object)
    output["rigid_n_observations"] = np.zeros(len(result), dtype=int)
    output["rigid_changed_points"] = np.full(len(result), "", dtype=object)
    output["is_rigid"] = np.zeros(len(result), dtype=bool)

    trial_column = (
        "trial_id" if "trial_id" in result.columns
        else "trial_id_" if "trial_id_" in result.columns
        else None
    )
    groups = result.groupby(trial_column, sort=False, dropna=False) if trial_column else [(None, result)]
    trusted_frames = 0

    for _, group in groups:
        for ordering_column in ("frame_idx", "frame", "time", "timestamp"):
            if ordering_column in group.columns:
                group = group.sort_values(ordering_column, kind="stable")
                break

        coordinate_columns = [
            f"{name}_cam_{axis}" for name in LANDMARKS for axis in ("x", "y")
        ]
        points = group[coordinate_columns].to_numpy(dtype=float).reshape(-1, 3, 2)
        finite = np.isfinite(points).all(axis=2)
        confidence_values = group[
            [confidence_columns[name] for name in LANDMARKS]
        ].apply(pd.to_numeric, errors="coerce").to_numpy()
        confident = finite & (confidence_values >= config.confidence_threshold)
        usable = finite & (
            confidence_values >= config.correction_confidence_threshold
        )
        low_probability = finite & (
            confidence_values < config.correction_confidence_threshold
        )

        seed = _trusted_seed_mask(points, confident, config)
        trusted_frames += int(seed.sum())
        group_indices = list(group.index)
        result_positions = result.index.get_indexer(group.index)
        if not seed.any():
            continue

        template = np.median(
            np.stack([_normalized_triangle(points[i]) for i in np.flatnonzero(seed)]),
            axis=0,
        )
        shape_positions = []
        normalized_shapes = {}
        for position in range(len(group)):
            if not finite[position].all():
                continue
            normalized = _normalized_triangle(points[position])
            shape_error = np.max(np.abs(
                _shape_signature(normalized) / _shape_signature(template) - 1.0
            ))
            if shape_error <= config.seed_shape_error:
                shape_positions.append(position)
                normalized_shapes[position] = normalized
        shape_positions = np.asarray(shape_positions, dtype=int)
        local_templates = []
        template_size = _side_lengths(template).mean()
        for position in range(len(group)):
            neighbors = shape_positions[
                np.abs(shape_positions - position) <= config.window_radius
            ]
            if not len(neighbors):
                local_templates.append(None)
                continue
            local_template = np.median(
                np.stack([normalized_shapes[neighbor] for neighbor in neighbors]),
                axis=0,
            )
            local_template *= template_size / _side_lengths(local_template).mean()
            local_templates.append(local_template)

        pose_measurements = np.full((len(group), 4), np.nan)
        measurement_mask = np.zeros(len(group), dtype=bool)
        for position in range(len(group)):
            anchors = np.flatnonzero(confident[position])
            if len(anchors) == 2:
                fitted = _fit_similarity(template, points[position], anchors)
            elif seed[position]:
                fitted = points[position]
            else:
                continue
            pose_measurements[position] = _pose_from_triangle(template, fitted)
            measurement_mask[position] = True

        smooth_pose = _smooth_pose_measurements(pose_measurements)
        positions = np.arange(len(group))
        before = np.maximum.accumulate(
            np.where(measurement_mask, positions, -100000)
        )
        after = np.minimum.accumulate(
            np.where(measurement_mask, positions, 100000)[::-1]
        )[::-1]
        near_measurement = np.minimum(positions - before, after - positions) <= config.window_radius

        for position, index in enumerate(group_indices):
            output_position = int(result_positions[position])
            observed = points[position]
            n_observations = int(usable[position].sum())

            if seed[position]:
                _write_output(output, output_position, observed, "raw_good_triangle", 0.0, 3, False)
                continue

            complete = finite[position].all()
            local_template = local_templates[position]
            shape_problem = False
            if complete and local_template is not None:
                shape_error = np.max(np.abs(
                    _shape_signature(observed) / _shape_signature(local_template) - 1.0
                ))
                shape_problem = shape_error > config.suspicious_shape_error
            candidate = not complete or low_probability[position].any() or shape_problem

            if not candidate:
                _write_output(
                    output, output_position, observed, "raw_good_triangle",
                    np.nan, n_observations, False
                )
                continue

            if not near_measurement[position] or not np.isfinite(smooth_pose[position]).all():
                _write_output(
                    output, output_position, observed,
                    "kalman_unresolved" if candidate else "raw_unverified_triangle",
                    np.nan, n_observations, False
                )
                continue

            pose_template = local_template if local_template is not None else template
            predicted = _triangle_from_pose(pose_template, smooth_pose[position])
            head_size = max(float(_side_lengths(predicted).mean()), 1e-6)
            residuals = np.full(len(LANDMARKS), np.inf)
            residuals[finite[position]] = np.linalg.norm(
                observed[finite[position]] - predicted[finite[position]], axis=1
            ) / head_size

            bad = ~finite[position] | low_probability[position]
            if shape_problem:
                bad |= residuals > config.point_outlier_error
                bad[int(np.argmax(residuals))] = True

            if not bad.any():
                _write_output(
                    output, output_position, observed, "raw_good_triangle",
                    float(np.sqrt(np.mean(residuals ** 2))),
                    n_observations, False
                )
                continue

            anchors = np.flatnonzero(usable[position] & ~bad)
            if not len(anchors):
                _write_output(
                    output, output_position, observed, "kalman_unresolved",
                    float(np.sqrt(np.mean(residuals ** 2))),
                    0, False
                )
                continue
            if len(anchors) >= 2:
                corrected_points = _fit_similarity(pose_template, observed, anchors)
            else:
                corrected_points = predicted + (
                    observed[anchors[0]] - predicted[anchors[0]]
                )
            changed = bad
            changed_points = ", ".join(
                LANDMARKS[i] for i in np.flatnonzero(changed)
            )
            corrected_error = float(np.sqrt(np.mean(
                np.sum((corrected_points - predicted) ** 2, axis=1)
            )) / head_size)
            method = (
                "kalman_missing" if not complete
                else "kalman_low_confidence" if low_probability[position].any()
                else "kalman_geometry"
            )
            _write_output(
                output, output_position, corrected_points, method,
                corrected_error, len(anchors), True, changed_points
            )

    if trusted_frames == 0:
        raise ValueError("No trustworthy high-confidence head frames were found")
    return result.join(pd.DataFrame(output, index=result.index))


def rigid_head_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return concise diagnostics shown by the UI."""
    corrected = df["is_rigid"].astype(bool)
    trial_column = (
        "trial_id" if "trial_id" in df.columns
        else "trial_id_" if "trial_id_" in df.columns
        else None
    )
    return {
        "total_rows": len(df),
        "corrected_rows": int(corrected.sum()),
        "unresolved_rows": int((df["rigid_method"] == "kalman_unresolved").sum()),
        "unavailable_rows": int((df["rigid_method"] == "unavailable").sum()),
        "corrected_trials": int(df.loc[corrected, trial_column].nunique()) if trial_column else int(corrected.any()),
    }
