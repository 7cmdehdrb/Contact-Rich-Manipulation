"""Reduction and physics checks for the 10-degree tilted cantilever test."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .measurement import WRENCH_COMPONENTS, WRENCH_UNITS


def expected_payload_wrench_per_kg(
    *,
    gravity_m_s2: float,
    tilt_rad: float,
    bending_lever_arm_m: float,
    payload_normal_offset_m: float,
) -> np.ndarray:
    """Return [Fx,Fy,Fz,Mx,My,Mz] per kg in the tilted sensor frame."""
    cosine = math.cos(tilt_rad)
    sine = math.sin(tilt_rad)
    return np.asarray(
        [
            gravity_m_s2 * cosine,
            -gravity_m_s2 * sine,
            0.0,
            gravity_m_s2 * bending_lever_arm_m * sine,
            gravity_m_s2 * bending_lever_arm_m * cosine,
            -gravity_m_s2 * payload_normal_offset_m * sine,
        ],
        dtype=float,
    )


def expected_tool_tare_wrench(
    *,
    tool_mass_kg: float,
    gravity_m_s2: float,
    tilt_rad: float,
    tool_center_z_m: float,
) -> np.ndarray:
    """Return the tool-only theoretical wrench at the Axia80 center."""
    cosine = math.cos(tilt_rad)
    sine = math.sin(tilt_rad)
    return tool_mass_kg * np.asarray(
        [
            gravity_m_s2 * cosine,
            -gravity_m_s2 * sine,
            0.0,
            gravity_m_s2 * tool_center_z_m * sine,
            gravity_m_s2 * tool_center_z_m * cosine,
            0.0,
        ],
        dtype=float,
    )


def summarize_tilted_wrenches(
    masses_g: Sequence[int],
    samples: np.ndarray,
    tare_samples: np.ndarray,
    *,
    gravity_m_s2: float,
    tilt_rad: float,
    bending_lever_arm_m: float,
    payload_normal_offset_m: float,
    expected_samples: np.ndarray | None = None,
    settled: Sequence[bool],
    settle_time_s: float,
    sample_max_relative_speed_m_s: Sequence[float],
    sample_max_relative_angular_speed_rad_s: Sequence[float],
    sample_max_position_error_m: Sequence[float],
    sample_max_abs_downhill_displacement_m: Sequence[float],
    sample_max_payload_tool_angle_deg: Sequence[float],
    sample_min_edge_margin_m: Sequence[float],
    sample_min_broad_normal_z: Sequence[float],
    sample_max_broad_normal_z: Sequence[float],
    sample_max_abs_sensor_axis_z: Sequence[float],
) -> list[dict[str, float | int | bool]]:
    """Produce one raw/tare/corrected row per tilted payload environment."""
    samples = np.asarray(samples, dtype=float)
    tare_samples = np.asarray(tare_samples, dtype=float)
    env_count = len(masses_g)
    if env_count == 0 or list(masses_g) != sorted(set(masses_g)):
        raise ValueError("masses_g must be strictly increasing and unique")
    if any(mass <= 0 for mass in masses_g):
        raise ValueError("masses_g must contain positive values")
    positive_scalars = {
        "gravity_m_s2": gravity_m_s2,
        "bending_lever_arm_m": bending_lever_arm_m,
        "payload_normal_offset_m": payload_normal_offset_m,
    }
    for name, value in positive_scalars.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(tilt_rad) or not 0.0 < abs(tilt_rad) < math.pi / 2.0:
        raise ValueError("tilt_rad must be finite, non-zero, and below 90 degrees")
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
        "sample_max_abs_downhill_displacement_m": sample_max_abs_downhill_displacement_m,
        "sample_max_payload_tool_angle_deg": sample_max_payload_tool_angle_deg,
        "sample_min_edge_margin_m": sample_min_edge_margin_m,
        "sample_min_broad_normal_z": sample_min_broad_normal_z,
        "sample_max_broad_normal_z": sample_max_broad_normal_z,
        "sample_max_abs_sensor_axis_z": sample_max_abs_sensor_axis_z,
    }
    for name, values in auxiliaries.items():
        if len(values) != env_count:
            raise ValueError(f"{name} must contain {env_count} values, got {len(values)}")
        if name != "settled" and not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"{name} contains non-finite values")

    if expected_samples is None:
        per_kg = expected_payload_wrench_per_kg(
            gravity_m_s2=gravity_m_s2,
            tilt_rad=tilt_rad,
            bending_lever_arm_m=bending_lever_arm_m,
            payload_normal_offset_m=payload_normal_offset_m,
        )
        expected_mean = np.asarray(masses_g, dtype=float)[:, None] * 1.0e-3 * per_kg
    else:
        expected_samples = np.asarray(expected_samples, dtype=float)
        if expected_samples.shape != samples.shape:
            raise ValueError(
                f"expected_samples must have shape {samples.shape}, got {expected_samples.shape}"
            )
        if not np.isfinite(expected_samples).all():
            raise ValueError("Expected wrench samples contain non-finite values")
        expected_mean = expected_samples.mean(axis=0)
    raw_mean = samples.mean(axis=0)
    raw_std = samples.std(axis=0, ddof=1)
    tare_mean = tare_samples.mean(axis=0)
    tare_std = tare_samples.std(axis=0, ddof=1)
    corrected = samples - tare_mean[None, :, :]
    corrected_mean = corrected.mean(axis=0)
    corrected_std = corrected.std(axis=0, ddof=1)

    rows: list[dict[str, float | int | bool]] = []
    for env_id, mass_g in enumerate(masses_g):
        expected = expected_mean[env_id]
        row: dict[str, float | int | bool] = {
            "env_id": env_id,
            "mass_g": int(mass_g),
            "mass_kg": mass_g * 1.0e-3,
            "tilt_deg": math.degrees(tilt_rad),
            "bending_lever_arm_m": bending_lever_arm_m,
            "payload_normal_offset_m": payload_normal_offset_m,
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
            "sample_max_abs_downhill_displacement_m": float(
                sample_max_abs_downhill_displacement_m[env_id]
            ),
            "sample_max_payload_tool_angle_deg": float(
                sample_max_payload_tool_angle_deg[env_id]
            ),
            "sample_min_edge_margin_m": float(sample_min_edge_margin_m[env_id]),
            "sample_min_broad_normal_z": float(sample_min_broad_normal_z[env_id]),
            "sample_max_broad_normal_z": float(sample_max_broad_normal_z[env_id]),
            "sample_max_abs_sensor_axis_z": float(
                sample_max_abs_sensor_axis_z[env_id]
            ),
        }
        for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
            row[f"expected_{name}_{unit}"] = float(expected[index])
            row[f"{name}_{unit}"] = float(corrected_mean[env_id, index])
            row[f"{name}_std_{unit}"] = float(corrected_std[env_id, index])
            row[f"raw_{name}_{unit}"] = float(raw_mean[env_id, index])
            row[f"raw_{name}_std_{unit}"] = float(raw_std[env_id, index])
            row[f"tare_{name}_{unit}"] = float(tare_mean[env_id, index])
            row[f"tare_{name}_std_{unit}"] = float(tare_std[env_id, index])
            row[f"{name}_error_{unit}"] = float(corrected_mean[env_id, index] - expected[index])
        row["measured_force_tilt_deg"] = math.degrees(
            math.atan2(-float(row["Fy_N"]), float(row["Fx_N"]))
        )
        row["measured_bending_tilt_deg"] = math.degrees(
            math.atan2(float(row["Mx_Nm"]), float(row["My_Nm"]))
        )
        row["measured_lever_from_My_Fx_m"] = float(row["My_Nm"]) / float(row["Fx_N"])
        row["measured_lever_from_Mx_Fy_m"] = float(row["Mx_Nm"]) / -float(row["Fy_N"])
        # This ratio is an effective diagnostic only: downhill COM displacement
        # also contributes -r_y*Fx to Mz, so it is not a pure height estimate.
        row["effective_Mz_Fy_ratio_m"] = float(row["Mz_Nm"]) / float(row["Fy_N"])
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Keep invalid distribution ratios inspectable instead of raising."""
    if abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def tilted_metrics(
    rows: Sequence[dict[str, object]],
    *,
    expected_tare: Sequence[float],
) -> dict[str, float | bool]:
    """Calculate per-axis slopes plus force/moment distribution diagnostics."""
    if len(rows) < 2:
        raise ValueError("At least two rows are required for regression")
    expected_tare_array = np.asarray(expected_tare, dtype=float)
    if expected_tare_array.shape != (6,) or not np.isfinite(expected_tare_array).all():
        raise ValueError("expected_tare must contain six finite values")

    mass_kg = np.asarray([float(row["mass_kg"]) for row in rows])
    metrics: dict[str, float | bool] = {}
    slopes: dict[str, float] = {}
    expected_slopes: dict[str, float] = {}
    for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS):
        measured = np.asarray([float(row[f"{name}_{unit}"]) for row in rows])
        expected = np.asarray([float(row[f"expected_{name}_{unit}"]) for row in rows])
        slope, intercept, r_squared = _linear_fit(mass_kg, measured)
        expected_slope, expected_intercept, _ = _linear_fit(mass_kg, expected)
        if float(np.ptp(expected)) > 1.0e-12:
            model_gain, model_intercept, model_r_squared = _linear_fit(expected, measured)
        else:
            model_gain = 0.0
            model_intercept = float(measured.mean())
            model_r_squared = 1.0
        key = name.lower()
        slopes[name] = slope
        expected_slopes[name] = expected_slope
        metrics[f"{key}_slope_{unit}_per_kg"] = slope
        metrics[f"expected_{key}_slope_{unit}_per_kg"] = expected_slope
        metrics[f"expected_{key}_intercept_{unit}"] = expected_intercept
        metrics[f"{key}_model_gain"] = model_gain
        metrics[f"{key}_model_intercept_{unit}"] = model_intercept
        metrics[f"{key}_model_r_squared"] = model_r_squared
        metrics[f"{key}_intercept_{unit}"] = intercept
        metrics[f"{key}_r_squared"] = r_squared
        metrics[f"{key}_rmse_{unit}"] = float(np.sqrt(np.mean((measured - expected) ** 2)))
        metrics[f"max_abs_{key}_error_{unit}"] = float(np.max(np.abs(measured - expected)))

    expected_force_magnitude = np.asarray(
        [float(row["mass_kg"]) for row in rows]
    ) * math.hypot(
        float(rows[0]["expected_Fx_N"]) / float(rows[0]["mass_kg"]),
        float(rows[0]["expected_Fy_N"]) / float(rows[0]["mass_kg"]),
    )
    measured_force_magnitude = np.asarray(
        [
            math.sqrt(
                float(row["Fx_N"]) ** 2
                + float(row["Fy_N"]) ** 2
                + float(row["Fz_N"]) ** 2
            )
            for row in rows
        ]
    )
    configured_tilt = float(rows[0]["tilt_deg"])
    force_tilt = math.degrees(math.atan2(-slopes["Fy"], slopes["Fx"]))
    bending_tilt = math.degrees(math.atan2(slopes["Mx"], slopes["My"]))
    expected_force_tilt = math.degrees(
        math.atan2(-expected_slopes["Fy"], expected_slopes["Fx"])
    )
    expected_bending_tilt = math.degrees(
        math.atan2(expected_slopes["Mx"], expected_slopes["My"])
    )
    lever_my_fx = _safe_ratio(slopes["My"], slopes["Fx"])
    lever_mx_fy = _safe_ratio(slopes["Mx"], -slopes["Fy"])
    effective_mz_fy_ratio = _safe_ratio(slopes["Mz"], slopes["Fy"])
    expected_lever = float(rows[0]["bending_lever_arm_m"])
    expected_normal_offset = float(rows[0]["payload_normal_offset_m"])
    metrics.update(
        {
            "force_resultant_rmse_N": float(
                np.sqrt(np.mean((measured_force_magnitude - expected_force_magnitude) ** 2))
            ),
            "max_abs_force_resultant_error_N": float(
                np.max(np.abs(measured_force_magnitude - expected_force_magnitude))
            ),
            "force_distribution_tilt_deg": force_tilt,
            "force_distribution_tilt_error_deg": force_tilt - configured_tilt,
            "expected_force_distribution_tilt_deg": expected_force_tilt,
            "force_distribution_model_error_deg": force_tilt - expected_force_tilt,
            "bending_distribution_tilt_deg": bending_tilt,
            "bending_distribution_tilt_error_deg": bending_tilt - configured_tilt,
            "expected_bending_distribution_tilt_deg": expected_bending_tilt,
            "bending_distribution_model_error_deg": bending_tilt - expected_bending_tilt,
            "lever_from_My_Fx_m": lever_my_fx,
            "lever_from_Mx_Fy_m": lever_mx_fy,
            "max_abs_lever_arm_error_m": max(
                abs(lever_my_fx - expected_lever),
                abs(lever_mx_fy - expected_lever),
            ),
            "effective_Mz_Fy_ratio_m": effective_mz_fy_ratio,
            "effective_Mz_Fy_ratio_minus_nominal_offset_m": (
                effective_mz_fy_ratio - expected_normal_offset
            ),
            "expected_broad_normal_z": math.cos(math.radians(configured_tilt)),
            "min_broad_normal_z": min(
                float(row["sample_min_broad_normal_z"]) for row in rows
            ),
            "max_broad_normal_z": max(
                float(row["sample_max_broad_normal_z"]) for row in rows
            ),
            "max_abs_sensor_axis_z": max(
                float(row["sample_max_abs_sensor_axis_z"]) for row in rows
            ),
            "max_sample_position_error_m": max(
                float(row["sample_max_position_error_m"]) for row in rows
            ),
            "max_abs_downhill_displacement_m": max(
                float(row["sample_max_abs_downhill_displacement_m"]) for row in rows
            ),
            "max_payload_tool_angle_deg": max(
                float(row["sample_max_payload_tool_angle_deg"]) for row in rows
            ),
            "min_payload_edge_margin_m": min(
                float(row["sample_min_edge_margin_m"]) for row in rows
            ),
            "all_settled": all(bool(row["settled"]) for row in rows),
            "primary_distributed_channels_monotonic": bool(
                all(
                    np.all(np.diff([float(row[column]) for row in rows]) > 0.0)
                    for column in ("Fx_N", "Mx_Nm", "My_Nm")
                )
                and np.all(np.diff([float(row["Fy_N"]) for row in rows]) < 0.0)
            ),
        }
    )

    for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
        tare = np.asarray([float(row[f"tare_{name}_{unit}"]) for row in rows])
        key = name.lower()
        metrics[f"mean_tare_{key}_{unit}"] = float(tare.mean())
        metrics[f"max_abs_tare_{key}_error_{unit}"] = float(
            np.max(np.abs(tare - expected_tare_array[index]))
        )
    return metrics


def evaluate_tilted_acceptance(
    metrics: dict[str, float | bool],
    *,
    expected_slopes: Sequence[float],
    position_tolerance_m: float,
    relative_slope_tolerance: float = 0.05,
    min_r_squared: float = 0.995,
    max_force_intercept_n: float = 0.02,
    max_moment_intercept_nm: float = 0.005,
    max_force_rmse_n: float = 0.01,
    max_moment_rmse_nm: float = 0.005,
    max_force_error_n: float = 0.02,
    max_moment_error_nm: float = 0.01,
    max_distribution_angle_error_deg: float = 0.5,
    max_lever_error_m: float = 0.005,
    max_tare_force_error_n: float = 0.02,
    max_tare_moment_error_nm: float = 0.01,
    max_payload_tool_angle_deg: float = 2.0,
    max_broad_normal_error: float = 0.002,
    max_abs_sensor_axis_z: float = 0.001,
) -> tuple[bool, list[str]]:
    """Validate that the 10-degree load is distributed as rigid-body statics predict."""
    expected = np.asarray(expected_slopes, dtype=float)
    if expected.shape != (6,) or not np.isfinite(expected).all():
        raise ValueError("expected_slopes must contain six finite values")
    failures: list[str] = []
    numeric = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric):
        failures.append("one or more acceptance metrics are non-finite")

    for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
        key = name.lower()
        slope = float(metrics[f"{key}_slope_{unit}_per_kg"])
        modeled_slope = float(metrics[f"expected_{key}_slope_{unit}_per_kg"])
        if abs(expected[index]) > 1.0e-12:
            if abs(slope - modeled_slope) > relative_slope_tolerance * abs(modeled_slope):
                failures.append(f"{name} slope does not match the tilted-static expectation")
            if float(metrics[f"{key}_model_r_squared"]) < min_r_squared:
                failures.append(f"{name} agreement with the sampled-pose model is too low")
        elif abs(slope) > 0.10:
            failures.append(f"nominally-zero {name} slope is too large")
        intercept_limit = max_force_intercept_n if index < 3 else max_moment_intercept_nm
        rmse_limit = max_force_rmse_n if index < 3 else max_moment_rmse_nm
        error_limit = max_force_error_n if index < 3 else max_moment_error_nm
        if abs(float(metrics[f"{key}_intercept_{unit}"])) > intercept_limit:
            failures.append(f"{name} intercept is too large")
        if float(metrics[f"{key}_rmse_{unit}"]) > rmse_limit:
            failures.append(f"{name} RMSE is too large")
        if float(metrics[f"max_abs_{key}_error_{unit}"]) > error_limit:
            failures.append(f"maximum {name} error is too large")

        tare_limit = max_tare_force_error_n if index < 3 else max_tare_moment_error_nm
        if float(metrics[f"max_abs_tare_{key}_error_{unit}"]) > tare_limit:
            failures.append(f"tool-only tare {name} is inaccurate")

    checks = (
        (
            float(metrics["force_resultant_rmse_N"]) <= max_force_rmse_n,
            "force resultant RMSE is too large",
        ),
        (
            float(metrics["max_abs_force_resultant_error_N"]) <= max_force_error_n,
            "maximum force resultant error is too large",
        ),
        (
            abs(float(metrics["force_distribution_model_error_deg"]))
            <= max_distribution_angle_error_deg,
            "force distribution angle does not match the sampled tool pose",
        ),
        (
            abs(float(metrics["bending_distribution_model_error_deg"]))
            <= max_distribution_angle_error_deg,
            "bending distribution angle does not match the sampled payload pose",
        ),
        (
            float(metrics["max_abs_lever_arm_error_m"]) <= max_lever_error_m,
            "measured bending lever arm is inaccurate",
        ),
        (
            bool(metrics["primary_distributed_channels_monotonic"]),
            "one or more primary distributed channels are not monotonic",
        ),
        (bool(metrics["all_settled"]), "one or more payloads were not stable"),
        (
            float(metrics["max_sample_position_error_m"]) <= position_tolerance_m,
            "payload position error exceeded the configured tolerance",
        ),
        (
            float(metrics["max_abs_downhill_displacement_m"]) <= position_tolerance_m,
            "payload slid too far down the inclined tool",
        ),
        (
            float(metrics["max_payload_tool_angle_deg"]) <= max_payload_tool_angle_deg,
            "payload tipped relative to the inclined tool",
        ),
        (
            float(metrics["min_payload_edge_margin_m"]) > 0.0,
            "payload reached or crossed the tool side edge",
        ),
        (
            abs(
                float(metrics["min_broad_normal_z"])
                - float(metrics["expected_broad_normal_z"])
            )
            <= max_broad_normal_error
            and abs(
                float(metrics["max_broad_normal_z"])
                - float(metrics["expected_broad_normal_z"])
            )
            <= max_broad_normal_error,
            "broad-face orientation does not match the configured tilt",
        ),
        (
            float(metrics["max_abs_sensor_axis_z"]) <= max_abs_sensor_axis_z,
            "cantilever axis is not sufficiently horizontal",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return not failures, failures
