"""Reduction, CSV, and acceptance checks for the wrist roll/tilt sweep."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .measurement import WRENCH_COMPONENTS, WRENCH_UNITS


def nominal_payload_wrench(
    *,
    mass_kg: float,
    gravity_m_s2: float,
    roll_deg: float,
    tilt_deg: float,
    bending_lever_arm_m: float,
    payload_normal_offset_m: float,
) -> np.ndarray:
    """Return the ideal payload reaction in the Axia80/tool frame."""
    values = (
        mass_kg,
        gravity_m_s2,
        roll_deg,
        tilt_deg,
        bending_lever_arm_m,
        payload_normal_offset_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nominal wrench inputs must be finite")
    if mass_kg <= 0.0 or gravity_m_s2 <= 0.0 or bending_lever_arm_m <= 0.0:
        raise ValueError("mass, gravity, and bending lever arm must be positive")
    if payload_normal_offset_m < 0.0:
        raise ValueError("payload_normal_offset_m must be non-negative")

    roll = math.radians(roll_deg)
    tilt = math.radians(tilt_deg)
    force = mass_kg * gravity_m_s2 * np.asarray(
        (
            math.cos(roll) * math.cos(tilt),
            -math.sin(roll) * math.cos(tilt),
            -math.sin(tilt),
        ),
        dtype=float,
    )
    position = np.asarray(
        (payload_normal_offset_m, 0.0, bending_lever_arm_m),
        dtype=float,
    )
    return np.concatenate((force, np.cross(position, force)))


def nominal_tool_tare_wrench(
    *,
    tool_mass_kg: float,
    gravity_m_s2: float,
    roll_deg: float,
    tilt_deg: float,
    tool_center_z_m: float,
) -> np.ndarray:
    """Return the ideal tool-only reaction for one commanded wrist pose."""
    payload_like = nominal_payload_wrench(
        mass_kg=tool_mass_kg,
        gravity_m_s2=gravity_m_s2,
        roll_deg=roll_deg,
        tilt_deg=tilt_deg,
        bending_lever_arm_m=tool_center_z_m,
        payload_normal_offset_m=0.0,
    )
    return payload_like


def _vector_angle_deg(measured: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(measured) * np.linalg.norm(expected))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.dot(measured, expected) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def summarize_wrist_angle_wrenches(
    angle_pairs_deg: Sequence[tuple[float, float]],
    payload_mass_g: int,
    samples: np.ndarray,
    tare_samples: np.ndarray,
    expected_samples: np.ndarray,
    expected_tare_samples: np.ndarray,
    *,
    gravity_m_s2: float,
    bending_lever_arm_m: float,
    payload_normal_offset_m: float,
    settled: Sequence[bool],
    settle_time_s: float,
    sample_mean_actual_roll_deg: Sequence[float],
    sample_mean_actual_tilt_deg: Sequence[float],
    sample_max_combined_inclination_deg: Sequence[float],
    sample_max_required_static_friction: Sequence[float],
    sample_max_relative_speed_m_s: Sequence[float],
    sample_max_relative_angular_speed_rad_s: Sequence[float],
    sample_max_position_error_m: Sequence[float],
    sample_max_tangent_displacement_m: Sequence[float],
    sample_max_payload_tool_angle_deg: Sequence[float],
    sample_min_width_edge_margin_m: Sequence[float],
    sample_min_length_edge_margin_m: Sequence[float],
) -> list[dict[str, float | int | bool]]:
    """Reduce raw windows to one inspectable row per commanded angle pair."""
    pairs = [(float(roll), float(tilt)) for roll, tilt in angle_pairs_deg]
    env_count = len(pairs)
    if env_count == 0 or len(set(pairs)) != env_count:
        raise ValueError("angle_pairs_deg must contain unique roll/tilt pairs")
    if payload_mass_g <= 0:
        raise ValueError("payload_mass_g must be positive")
    arrays = {
        "samples": np.asarray(samples, dtype=float),
        "tare_samples": np.asarray(tare_samples, dtype=float),
        "expected_samples": np.asarray(expected_samples, dtype=float),
        "expected_tare_samples": np.asarray(expected_tare_samples, dtype=float),
    }
    for name, array in arrays.items():
        if array.ndim != 3 or array.shape[1:] != (env_count, 6):
            raise ValueError(
                f"{name} must have shape (samples, {env_count}, 6), got {array.shape}"
            )
        if array.shape[0] < 2 or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain at least two finite samples")
    if arrays["samples"].shape != arrays["expected_samples"].shape:
        raise ValueError("samples and expected_samples must have matching shapes")
    if arrays["tare_samples"].shape != arrays["expected_tare_samples"].shape:
        raise ValueError("tare samples and expected tare samples must have matching shapes")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    auxiliaries = {
        "settled": settled,
        "sample_mean_actual_roll_deg": sample_mean_actual_roll_deg,
        "sample_mean_actual_tilt_deg": sample_mean_actual_tilt_deg,
        "sample_max_combined_inclination_deg": sample_max_combined_inclination_deg,
        "sample_max_required_static_friction": sample_max_required_static_friction,
        "sample_max_relative_speed_m_s": sample_max_relative_speed_m_s,
        "sample_max_relative_angular_speed_rad_s": sample_max_relative_angular_speed_rad_s,
        "sample_max_position_error_m": sample_max_position_error_m,
        "sample_max_tangent_displacement_m": sample_max_tangent_displacement_m,
        "sample_max_payload_tool_angle_deg": sample_max_payload_tool_angle_deg,
        "sample_min_width_edge_margin_m": sample_min_width_edge_margin_m,
        "sample_min_length_edge_margin_m": sample_min_length_edge_margin_m,
    }
    for name, values in auxiliaries.items():
        if len(values) != env_count:
            raise ValueError(f"{name} must contain {env_count} values, got {len(values)}")
        if name != "settled" and not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"{name} contains non-finite values")

    raw_mean = arrays["samples"].mean(axis=0)
    raw_std = arrays["samples"].std(axis=0, ddof=1)
    tare_mean = arrays["tare_samples"].mean(axis=0)
    tare_std = arrays["tare_samples"].std(axis=0, ddof=1)
    corrected = arrays["samples"] - tare_mean[None, :, :]
    corrected_mean = corrected.mean(axis=0)
    corrected_std = corrected.std(axis=0, ddof=1)
    expected_mean = arrays["expected_samples"].mean(axis=0)
    expected_tare_mean = arrays["expected_tare_samples"].mean(axis=0)
    mass_kg = payload_mass_g * 1.0e-3

    rows: list[dict[str, float | int | bool]] = []
    for env_id, (roll_deg, tilt_deg) in enumerate(pairs):
        nominal = nominal_payload_wrench(
            mass_kg=mass_kg,
            gravity_m_s2=gravity_m_s2,
            roll_deg=roll_deg,
            tilt_deg=tilt_deg,
            bending_lever_arm_m=bending_lever_arm_m,
            payload_normal_offset_m=payload_normal_offset_m,
        )
        actual_roll_deg = float(sample_mean_actual_roll_deg[env_id])
        actual_tilt_deg = float(sample_mean_actual_tilt_deg[env_id])
        force_error = corrected_mean[env_id, :3] - expected_mean[env_id, :3]
        moment_error = corrected_mean[env_id, 3:] - expected_mean[env_id, 3:]
        row: dict[str, float | int | bool] = {
            "env_id": env_id,
            "mass_g": int(payload_mass_g),
            "mass_kg": mass_kg,
            "commanded_roll_deg": roll_deg,
            "commanded_tilt_deg": tilt_deg,
            "actual_roll_deg": actual_roll_deg,
            "actual_tilt_deg": actual_tilt_deg,
            "sample_max_combined_inclination_deg": float(
                sample_max_combined_inclination_deg[env_id]
            ),
            "sample_max_required_static_friction": float(
                sample_max_required_static_friction[env_id]
            ),
            "bending_lever_arm_m": float(bending_lever_arm_m),
            "payload_normal_offset_m": float(payload_normal_offset_m),
            "settled": bool(settled[env_id]),
            "settle_time_s": float(settle_time_s),
            "sample_count": int(arrays["samples"].shape[0]),
            "tare_sample_count": int(arrays["tare_samples"].shape[0]),
            "sample_max_relative_speed_m_s": float(
                sample_max_relative_speed_m_s[env_id]
            ),
            "sample_max_relative_angular_speed_rad_s": float(
                sample_max_relative_angular_speed_rad_s[env_id]
            ),
            "sample_max_position_error_m": float(sample_max_position_error_m[env_id]),
            "sample_max_tangent_displacement_m": float(
                sample_max_tangent_displacement_m[env_id]
            ),
            "sample_max_payload_tool_angle_deg": float(
                sample_max_payload_tool_angle_deg[env_id]
            ),
            "sample_min_width_edge_margin_m": float(
                sample_min_width_edge_margin_m[env_id]
            ),
            "sample_min_length_edge_margin_m": float(
                sample_min_length_edge_margin_m[env_id]
            ),
            "force_direction_error_deg": _vector_angle_deg(
                corrected_mean[env_id, :3], expected_mean[env_id, :3]
            ),
            "moment_direction_error_deg": _vector_angle_deg(
                corrected_mean[env_id, 3:], expected_mean[env_id, 3:]
            ),
            "force_vector_error_N": float(np.linalg.norm(force_error)),
            "moment_vector_error_Nm": float(np.linalg.norm(moment_error)),
        }
        for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
            row[f"nominal_expected_{name}_{unit}"] = float(nominal[index])
            row[f"expected_{name}_{unit}"] = float(expected_mean[env_id, index])
            row[f"{name}_{unit}"] = float(corrected_mean[env_id, index])
            row[f"{name}_std_{unit}"] = float(corrected_std[env_id, index])
            row[f"{name}_error_{unit}"] = float(
                corrected_mean[env_id, index] - expected_mean[env_id, index]
            )
            row[f"raw_{name}_{unit}"] = float(raw_mean[env_id, index])
            row[f"raw_{name}_std_{unit}"] = float(raw_std[env_id, index])
            row[f"tare_{name}_{unit}"] = float(tare_mean[env_id, index])
            row[f"tare_{name}_std_{unit}"] = float(tare_std[env_id, index])
            row[f"expected_tare_{name}_{unit}"] = float(
                expected_tare_mean[env_id, index]
            )
            row[f"tare_{name}_error_{unit}"] = float(
                tare_mean[env_id, index] - expected_tare_mean[env_id, index]
            )
        rows.append(row)
    return rows


def wrist_angle_metrics(
    rows: Sequence[dict[str, object]],
) -> dict[str, float | int | bool]:
    """Aggregate pointwise model agreement and contact-stability metrics."""
    if len(rows) < 4:
        raise ValueError("At least four angle-grid rows are required")
    force_errors = np.asarray(
        [[float(row[f"{name}_error_N"]) for name in ("Fx", "Fy", "Fz")] for row in rows]
    )
    moment_errors = np.asarray(
        [
            [float(row[f"{name}_error_Nm"]) for name in ("Mx", "My", "Mz")]
            for row in rows
        ]
    )
    tare_force_errors = np.asarray(
        [
            [float(row[f"tare_{name}_error_N"]) for name in ("Fx", "Fy", "Fz")]
            for row in rows
        ]
    )
    tare_moment_errors = np.asarray(
        [
            [float(row[f"tare_{name}_error_Nm"]) for name in ("Mx", "My", "Mz")]
            for row in rows
        ]
    )
    rolls = np.asarray([float(row["commanded_roll_deg"]) for row in rows])
    tilts = np.asarray([float(row["commanded_tilt_deg"]) for row in rows])
    actual_rolls = np.asarray([float(row["actual_roll_deg"]) for row in rows])
    actual_tilts = np.asarray([float(row["actual_tilt_deg"]) for row in rows])
    unique_rolls = np.unique(rolls)
    unique_tilts = np.unique(tilts)
    complete_grid = len(rows) == len(unique_rolls) * len(unique_tilts)
    metrics: dict[str, float | int | bool] = {
        "num_environments": len(rows),
        "num_roll_angles": int(len(unique_rolls)),
        "num_tilt_angles": int(len(unique_tilts)),
        "roll_span_deg": float(np.ptp(unique_rolls)),
        "tilt_span_deg": float(np.ptp(unique_tilts)),
        "complete_angle_grid": bool(complete_grid),
        "force_component_rmse_N": float(np.sqrt(np.mean(force_errors**2))),
        "moment_component_rmse_Nm": float(np.sqrt(np.mean(moment_errors**2))),
        "max_abs_force_component_error_N": float(np.max(np.abs(force_errors))),
        "max_abs_moment_component_error_Nm": float(np.max(np.abs(moment_errors))),
        "max_force_vector_error_N": max(float(row["force_vector_error_N"]) for row in rows),
        "max_moment_vector_error_Nm": max(
            float(row["moment_vector_error_Nm"]) for row in rows
        ),
        "max_force_direction_error_deg": max(
            float(row["force_direction_error_deg"]) for row in rows
        ),
        "max_moment_direction_error_deg": max(
            float(row["moment_direction_error_deg"]) for row in rows
        ),
        "max_abs_tare_force_component_error_N": float(
            np.max(np.abs(tare_force_errors))
        ),
        "max_abs_tare_moment_component_error_Nm": float(
            np.max(np.abs(tare_moment_errors))
        ),
        "max_abs_roll_tracking_error_deg": float(np.max(np.abs(actual_rolls - rolls))),
        "max_abs_tilt_tracking_error_deg": float(np.max(np.abs(actual_tilts - tilts))),
        "max_combined_inclination_deg": max(
            float(row["sample_max_combined_inclination_deg"]) for row in rows
        ),
        "max_required_static_friction": max(
            float(row["sample_max_required_static_friction"]) for row in rows
        ),
        "all_settled": all(bool(row["settled"]) for row in rows),
        "max_sample_position_error_m": max(
            float(row["sample_max_position_error_m"]) for row in rows
        ),
        "max_tangent_displacement_m": max(
            float(row["sample_max_tangent_displacement_m"]) for row in rows
        ),
        "max_payload_tool_angle_deg": max(
            float(row["sample_max_payload_tool_angle_deg"]) for row in rows
        ),
        "min_width_edge_margin_m": min(
            float(row["sample_min_width_edge_margin_m"]) for row in rows
        ),
        "min_length_edge_margin_m": min(
            float(row["sample_min_length_edge_margin_m"]) for row in rows
        ),
    }
    for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS):
        values = np.asarray([float(row[f"{name}_{unit}"]) for row in rows])
        expected = np.asarray([float(row[f"expected_{name}_{unit}"]) for row in rows])
        metrics[f"{name.lower()}_range_{unit}"] = float(np.ptp(values))
        metrics[f"expected_{name.lower()}_range_{unit}"] = float(np.ptp(expected))
    return metrics


def evaluate_wrist_angle_acceptance(
    metrics: dict[str, float | int | bool],
    *,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    static_friction: float,
    max_force_component_rmse_n: float = 0.015,
    max_moment_component_rmse_nm: float = 0.004,
    max_force_component_error_n: float = 0.03,
    max_moment_component_error_nm: float = 0.01,
    max_force_vector_error_n: float = 0.04,
    max_moment_vector_error_nm: float = 0.015,
    max_direction_error_deg: float = 0.75,
    max_tare_force_error_n: float = 0.03,
    max_tare_moment_error_nm: float = 0.01,
    max_joint_tracking_error_deg: float = 1.0,
) -> tuple[bool, list[str]]:
    """Return a strict pointwise verdict for the two-dimensional angle sweep."""
    if not math.isfinite(static_friction) or static_friction <= 0.0:
        raise ValueError("static_friction must be finite and positive")
    failures: list[str] = []
    numeric = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric):
        failures.append("one or more acceptance metrics are non-finite")
    checks = (
        (bool(metrics["complete_angle_grid"]), "roll/tilt combinations do not form a complete grid"),
        (int(metrics["num_roll_angles"]) >= 2, "at least two roll angles are required"),
        (int(metrics["num_tilt_angles"]) >= 2, "at least two tilt angles are required"),
        (
            float(metrics["force_component_rmse_N"]) <= max_force_component_rmse_n,
            "force component RMSE is too large",
        ),
        (
            float(metrics["moment_component_rmse_Nm"]) <= max_moment_component_rmse_nm,
            "moment component RMSE is too large",
        ),
        (
            float(metrics["max_abs_force_component_error_N"])
            <= max_force_component_error_n,
            "maximum force component error is too large",
        ),
        (
            float(metrics["max_abs_moment_component_error_Nm"])
            <= max_moment_component_error_nm,
            "maximum moment component error is too large",
        ),
        (
            float(metrics["max_force_vector_error_N"]) <= max_force_vector_error_n,
            "maximum force-vector error is too large",
        ),
        (
            float(metrics["max_moment_vector_error_Nm"]) <= max_moment_vector_error_nm,
            "maximum moment-vector error is too large",
        ),
        (
            float(metrics["max_force_direction_error_deg"]) <= max_direction_error_deg,
            "force-vector direction does not match the sampled pose",
        ),
        (
            float(metrics["max_moment_direction_error_deg"]) <= max_direction_error_deg,
            "moment-vector direction does not match the sampled pose",
        ),
        (
            float(metrics["max_abs_tare_force_component_error_N"])
            <= max_tare_force_error_n,
            "tool-only tare force is inaccurate",
        ),
        (
            float(metrics["max_abs_tare_moment_component_error_Nm"])
            <= max_tare_moment_error_nm,
            "tool-only tare moment is inaccurate",
        ),
        (
            float(metrics["max_abs_roll_tracking_error_deg"])
            <= max_joint_tracking_error_deg,
            "actual roll differs too much from the command",
        ),
        (
            float(metrics["max_abs_tilt_tracking_error_deg"])
            <= max_joint_tracking_error_deg,
            "actual tilt differs too much from the command",
        ),
        (
            float(metrics["max_required_static_friction"]) < static_friction,
            "configured static friction is insufficient for the sampled incline",
        ),
        (bool(metrics["all_settled"]), "one or more payloads were not stable"),
        (
            float(metrics["max_sample_position_error_m"]) <= position_tolerance_m,
            "payload position error exceeded the configured tolerance",
        ),
        (
            float(metrics["max_tangent_displacement_m"]) <= position_tolerance_m,
            "payload slid too far across the tool surface",
        ),
        (
            float(metrics["max_payload_tool_angle_deg"]) <= orientation_tolerance_deg,
            "payload tipped relative to the tool",
        ),
        (
            float(metrics["min_width_edge_margin_m"]) > 0.0,
            "payload reached or crossed a tool width edge",
        ),
        (
            float(metrics["min_length_edge_margin_m"]) > 0.0,
            "payload reached or crossed a tool length edge",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return not failures, failures


def write_wrist_angle_sample_csv(
    path: Path,
    angle_pairs_deg: Sequence[tuple[float, float]],
    payload_mass_g: int,
    samples: np.ndarray,
    tare_mean: np.ndarray,
    expected_samples: np.ndarray,
    *,
    dt_s: float,
) -> None:
    """Write every sample with its commanded angles and pose-based expectation."""
    pairs = list(angle_pairs_deg)
    samples = np.asarray(samples, dtype=float)
    tare_mean = np.asarray(tare_mean, dtype=float)
    expected_samples = np.asarray(expected_samples, dtype=float)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if samples.ndim != 3 or samples.shape[1:] != (len(pairs), 6):
        raise ValueError("samples do not match the angle grid")
    if samples.shape != expected_samples.shape or tare_mean.shape != (len(pairs), 6):
        raise ValueError("expected samples or tare mean do not match the raw samples")
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if not all(np.isfinite(array).all() for array in (samples, tare_mean, expected_samples)):
        raise ValueError("sample CSV inputs contain non-finite values")

    corrected_names = [f"{name}_{unit}" for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS)]
    raw_names = [f"raw_{name}_{unit}" for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS)]
    expected_names = [
        f"expected_{name}_{unit}" for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_index",
                "time_s",
                "env_id",
                "mass_g",
                "commanded_roll_deg",
                "commanded_tilt_deg",
                *corrected_names,
                *raw_names,
                *expected_names,
            ]
        )
        for sample_index, sample in enumerate(samples):
            corrected = sample - tare_mean
            for env_id, (roll_deg, tilt_deg) in enumerate(pairs):
                writer.writerow(
                    [
                        sample_index,
                        sample_index * dt_s,
                        env_id,
                        payload_mass_g,
                        roll_deg,
                        tilt_deg,
                        *corrected[env_id].tolist(),
                        *sample[env_id].tolist(),
                        *expected_samples[sample_index, env_id].tolist(),
                    ]
                )


def read_wrist_angle_summary_csv(path: Path) -> list[dict[str, float | int | bool]]:
    """Read a fixed-mass angle summary without imposing unique masses."""
    rows: list[dict[str, float | int | bool]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, float | int | bool] = {}
            for key, value in raw.items():
                if key in ("env_id", "mass_g", "sample_count", "tare_sample_count"):
                    row[key] = int(value)
                elif key == "settled":
                    row[key] = value.lower() == "true"
                else:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"Non-finite value in {path}: {key}={value}")
                    row[key] = number
            rows.append(row)
    if not rows:
        raise ValueError(f"Summary CSV is empty: {path}")
    pairs = [
        (float(row["commanded_roll_deg"]), float(row["commanded_tilt_deg"]))
        for row in rows
    ]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Summary roll/tilt pairs must be unique")
    if len({int(row["mass_g"]) for row in rows}) != 1:
        raise ValueError("All angle-sweep rows must use one common payload mass")
    return rows


__all__ = [
    "evaluate_wrist_angle_acceptance",
    "nominal_payload_wrench",
    "nominal_tool_tare_wrench",
    "read_wrist_angle_summary_csv",
    "summarize_wrist_angle_wrenches",
    "write_wrist_angle_sample_csv",
    "wrist_angle_metrics",
]
