"""Reduction and acceptance checks for the thin-face cantilever experiment."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .measurement import WRENCH_COMPONENTS, WRENCH_UNITS


def summarize_cantilever_wrenches(
    masses_g: Sequence[int],
    samples: np.ndarray,
    tare_samples: np.ndarray,
    *,
    gravity_m_s2: float,
    bending_lever_arm_m: float,
    settled: Sequence[bool],
    settle_time_s: float,
    sample_max_relative_speed_m_s: Sequence[float],
    sample_max_relative_angular_speed_rad_s: Sequence[float],
    sample_max_position_error_m: Sequence[float],
    sample_min_broad_normal_z: Sequence[float],
    sample_max_abs_sensor_axis_z: Sequence[float],
) -> list[dict[str, float | int | bool]]:
    """Reduce raw windows to per-mass raw and tare-corrected wrench rows."""
    samples = np.asarray(samples, dtype=float)
    tare_samples = np.asarray(tare_samples, dtype=float)
    env_count = len(masses_g)
    if env_count == 0 or list(masses_g) != sorted(set(masses_g)):
        raise ValueError("masses_g must be strictly increasing and unique")
    if any(mass <= 0 for mass in masses_g):
        raise ValueError("masses_g must contain positive values")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    if not math.isfinite(bending_lever_arm_m) or bending_lever_arm_m <= 0.0:
        raise ValueError("bending_lever_arm_m must be finite and positive")
    if samples.ndim != 3 or samples.shape[1:] != (env_count, 6):
        raise ValueError(f"samples must have shape (S, {env_count}, 6), got {samples.shape}")
    if tare_samples.ndim != 3 or tare_samples.shape[1:] != (env_count, 6):
        raise ValueError(
            f"tare_samples must have shape (T, {env_count}, 6), got {tare_samples.shape}"
        )
    if samples.shape[0] < 2 or tare_samples.shape[0] < 2:
        raise ValueError("At least two payload and tare samples are required")
    if not np.isfinite(samples).all() or not np.isfinite(tare_samples).all():
        raise ValueError("Wrench samples contain non-finite values")

    auxiliaries = {
        "settled": settled,
        "sample_max_relative_speed_m_s": sample_max_relative_speed_m_s,
        "sample_max_relative_angular_speed_rad_s": sample_max_relative_angular_speed_rad_s,
        "sample_max_position_error_m": sample_max_position_error_m,
        "sample_min_broad_normal_z": sample_min_broad_normal_z,
        "sample_max_abs_sensor_axis_z": sample_max_abs_sensor_axis_z,
    }
    for name, values in auxiliaries.items():
        if len(values) != env_count:
            raise ValueError(f"{name} must contain {env_count} values, got {len(values)}")
        if name != "settled" and not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"{name} contains non-finite values")

    raw_mean = samples.mean(axis=0)
    raw_std = samples.std(axis=0, ddof=1)
    tare_mean = tare_samples.mean(axis=0)
    tare_std = tare_samples.std(axis=0, ddof=1)
    corrected = samples - tare_mean[None, :, :]
    corrected_mean = corrected.mean(axis=0)
    corrected_std = corrected.std(axis=0, ddof=1)

    rows: list[dict[str, float | int | bool]] = []
    for env_id, mass_g in enumerate(masses_g):
        expected_fx = mass_g * 1.0e-3 * gravity_m_s2
        expected_my = expected_fx * bending_lever_arm_m
        row: dict[str, float | int | bool] = {
            "env_id": env_id,
            "mass_g": int(mass_g),
            "mass_kg": mass_g * 1.0e-3,
            "bending_lever_arm_m": bending_lever_arm_m,
            "expected_Fx_N": expected_fx,
            "expected_My_Nm": expected_my,
            "settled": bool(settled[env_id]),
            "settle_time_s": float(settle_time_s),
            "sample_count": int(samples.shape[0]),
            "tare_sample_count": int(tare_samples.shape[0]),
            "sample_max_relative_speed_m_s": float(
                sample_max_relative_speed_m_s[env_id]
            ),
            "sample_max_relative_angular_speed_rad_s": float(
                sample_max_relative_angular_speed_rad_s[env_id]
            ),
            "sample_max_position_error_m": float(sample_max_position_error_m[env_id]),
            "sample_min_broad_normal_z": float(sample_min_broad_normal_z[env_id]),
            "sample_max_abs_sensor_axis_z": float(
                sample_max_abs_sensor_axis_z[env_id]
            ),
        }
        for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
            row[f"{name}_{unit}"] = float(corrected_mean[env_id, index])
            row[f"{name}_std_{unit}"] = float(corrected_std[env_id, index])
            row[f"raw_{name}_{unit}"] = float(raw_mean[env_id, index])
            row[f"raw_{name}_std_{unit}"] = float(raw_std[env_id, index])
            row[f"tare_{name}_{unit}"] = float(tare_mean[env_id, index])
            row[f"tare_{name}_std_{unit}"] = float(tare_std[env_id, index])
        row["Fx_error_N"] = float(corrected_mean[env_id, 0] - expected_fx)
        row["Fx_error_percent"] = float(
            100.0 * (corrected_mean[env_id, 0] - expected_fx) / expected_fx
        )
        row["My_error_Nm"] = float(corrected_mean[env_id, 4] - expected_my)
        row["My_error_percent"] = float(
            100.0 * (corrected_mean[env_id, 4] - expected_my) / expected_my
        )
        row["measured_lever_arm_m"] = float(
            corrected_mean[env_id, 4] / corrected_mean[env_id, 0]
        )
        rows.append(row)
    return rows


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum((y - fitted) ** 2)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return float(slope), float(intercept), float(r_squared)


def cantilever_metrics(
    rows: Sequence[dict[str, object]],
    *,
    expected_tare_fx_n: float,
    expected_tare_my_nm: float,
) -> dict[str, float | bool]:
    """Calculate normal-force, bending-moment, tare, and cross-axis metrics."""
    if len(rows) < 2:
        raise ValueError("At least two rows are required for regression")
    mass_kg = np.asarray([float(row["mass_kg"]) for row in rows])
    fx = np.asarray([float(row["Fx_N"]) for row in rows])
    my = np.asarray([float(row["My_Nm"]) for row in rows])
    expected_fx = np.asarray([float(row["expected_Fx_N"]) for row in rows])
    expected_my = np.asarray([float(row["expected_My_Nm"]) for row in rows])
    measured_lever = np.asarray([float(row["measured_lever_arm_m"]) for row in rows])
    expected_lever = np.asarray([float(row["bending_lever_arm_m"]) for row in rows])
    fx_slope, fx_intercept, fx_r_squared = _linear_fit(mass_kg, fx)
    my_slope, my_intercept, my_r_squared = _linear_fit(mass_kg, my)

    tare_fx = np.asarray([float(row["tare_Fx_N"]) for row in rows])
    tare_my = np.asarray([float(row["tare_My_Nm"]) for row in rows])
    metrics: dict[str, float | bool] = {
        "fx_slope_N_per_kg": fx_slope,
        "fx_intercept_N": fx_intercept,
        "fx_r_squared": fx_r_squared,
        "fx_rmse_N": float(np.sqrt(np.mean((fx - expected_fx) ** 2))),
        "max_abs_fx_error_N": float(np.max(np.abs(fx - expected_fx))),
        "fx_monotonic": bool(np.all(np.diff(fx) > 0.0)),
        "my_slope_Nm_per_kg": my_slope,
        "my_intercept_Nm": my_intercept,
        "my_r_squared": my_r_squared,
        "my_rmse_Nm": float(np.sqrt(np.mean((my - expected_my) ** 2))),
        "max_abs_my_error_Nm": float(np.max(np.abs(my - expected_my))),
        "my_monotonic": bool(np.all(np.diff(my) > 0.0)),
        "max_abs_lever_arm_error_m": float(
            np.max(np.abs(measured_lever - expected_lever))
        ),
        "max_cross_axis_force_N": float(
            max(abs(float(row[name])) for name in ("Fy_N", "Fz_N") for row in rows)
        ),
        "max_cross_axis_moment_Nm": float(
            max(abs(float(row[name])) for name in ("Mx_Nm", "Mz_Nm") for row in rows)
        ),
        "all_settled": all(bool(row["settled"]) for row in rows),
        "max_sample_position_error_m": max(
            float(row["sample_max_position_error_m"]) for row in rows
        ),
        "min_broad_normal_z": min(
            float(row["sample_min_broad_normal_z"]) for row in rows
        ),
        "max_abs_sensor_axis_z": max(
            float(row["sample_max_abs_sensor_axis_z"]) for row in rows
        ),
        "mean_tare_fx_N": float(tare_fx.mean()),
        "max_abs_tare_fx_error_N": float(np.max(np.abs(tare_fx - expected_tare_fx_n))),
        "tare_fx_span_N": float(np.ptp(tare_fx)),
        "mean_tare_my_Nm": float(tare_my.mean()),
        "max_abs_tare_my_error_Nm": float(
            np.max(np.abs(tare_my - expected_tare_my_nm))
        ),
        "tare_my_span_Nm": float(np.ptp(tare_my)),
    }
    return metrics


def evaluate_cantilever_acceptance(
    metrics: dict[str, float | bool],
    *,
    gravity_m_s2: float,
    bending_lever_arm_m: float,
    relative_slope_tolerance: float = 0.05,
    min_r_squared: float = 0.995,
    max_abs_fx_intercept_n: float = 0.02,
    max_fx_rmse_n: float = 0.01,
    max_abs_fx_error_n: float = 0.02,
    max_abs_my_intercept_nm: float = 0.005,
    max_my_rmse_nm: float = 0.005,
    max_abs_my_error_nm: float = 0.01,
    max_abs_lever_arm_error_m: float = 0.005,
    max_abs_tare_fx_error_n: float = 0.02,
    max_abs_tare_my_error_nm: float = 0.01,
    max_cross_axis_force_n: float = 0.10,
    max_cross_axis_moment_nm: float = 0.01,
    min_broad_normal_z: float = 0.999,
    max_abs_sensor_axis_z: float = 0.001,
) -> tuple[bool, list[str]]:
    """Return a strict verdict for force transmission and bending response."""
    failures: list[str] = []
    numeric = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric):
        failures.append("one or more acceptance metrics are non-finite")

    expected_my_slope = gravity_m_s2 * bending_lever_arm_m
    checks = (
        (
            abs(float(metrics["fx_slope_N_per_kg"]) - gravity_m_s2)
            <= relative_slope_tolerance * gravity_m_s2,
            "Fx slope does not match gravity",
        ),
        (
            abs(float(metrics["my_slope_Nm_per_kg"]) - expected_my_slope)
            <= relative_slope_tolerance * expected_my_slope,
            "My slope does not match gravity times the bending lever arm",
        ),
        (float(metrics["fx_r_squared"]) >= min_r_squared, "Fx R^2 is too low"),
        (float(metrics["my_r_squared"]) >= min_r_squared, "My R^2 is too low"),
        (
            abs(float(metrics["fx_intercept_N"])) <= max_abs_fx_intercept_n,
            "Fx intercept is too large",
        ),
        (
            float(metrics["fx_rmse_N"]) <= max_fx_rmse_n,
            "Fx RMSE is too large",
        ),
        (
            float(metrics["max_abs_fx_error_N"]) <= max_abs_fx_error_n,
            "maximum |Fx-m*g| is too large",
        ),
        (
            abs(float(metrics["my_intercept_Nm"])) <= max_abs_my_intercept_nm,
            "My intercept is too large",
        ),
        (
            float(metrics["my_rmse_Nm"]) <= max_my_rmse_nm,
            "My RMSE is too large",
        ),
        (
            float(metrics["max_abs_my_error_Nm"]) <= max_abs_my_error_nm,
            "maximum My bending error is too large",
        ),
        (
            float(metrics["max_abs_lever_arm_error_m"]) <= max_abs_lever_arm_error_m,
            "measured My/Fx lever arm is inaccurate",
        ),
        (bool(metrics["fx_monotonic"]), "tare-corrected Fx is not strictly increasing"),
        (bool(metrics["my_monotonic"]), "tare-corrected My is not strictly increasing"),
        (
            float(metrics["max_abs_tare_fx_error_N"]) <= max_abs_tare_fx_error_n,
            "tool-only tare Fx is inaccurate",
        ),
        (
            float(metrics["max_abs_tare_my_error_Nm"]) <= max_abs_tare_my_error_nm,
            "tool-only tare My is inaccurate",
        ),
        (
            float(metrics["max_cross_axis_force_N"]) <= max_cross_axis_force_n,
            "cross-axis force is too large",
        ),
        (
            float(metrics["max_cross_axis_moment_Nm"]) <= max_cross_axis_moment_nm,
            "cross-axis moment is too large",
        ),
        (bool(metrics["all_settled"]), "one or more payloads were not stable"),
        (
            float(metrics["min_broad_normal_z"]) >= min_broad_normal_z,
            "tool broad-face normal is not sufficiently upward",
        ),
        (
            float(metrics["max_abs_sensor_axis_z"]) <= max_abs_sensor_axis_z,
            "sensor/cantilever axis is not sufficiently horizontal",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return not failures, failures
