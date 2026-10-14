"""Pure data reduction, CSV, and acceptance checks for the payload experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
WRENCH_UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")


def default_masses_g(start_g: int = 10, stop_g: int = 500, step_g: int = 10) -> list[int]:
    """Return an inclusive, strictly increasing payload sequence in grams."""
    if start_g <= 0 or stop_g < start_g or step_g <= 0:
        raise ValueError("Mass range requires 0 < start_g <= stop_g and step_g > 0")
    masses = list(range(start_g, stop_g + 1, step_g))
    if masses[-1] != stop_g:
        raise ValueError("stop_g must be reachable exactly from start_g using step_g")
    return masses


def summarize_wrenches(
    masses_g: Sequence[int],
    samples: np.ndarray,
    tare_samples: np.ndarray,
    *,
    gravity_m_s2: float,
    settled: Sequence[bool],
    settle_time_s: float,
    sample_max_relative_speed_m_s: Sequence[float],
    sample_max_relative_angular_speed_rad_s: Sequence[float],
    tool_normal_z: Sequence[float],
) -> list[dict[str, float | int | bool]]:
    """Reduce ``(samples, envs, 6)`` arrays to one inspectable row per mass."""
    samples = np.asarray(samples, dtype=float)
    tare_samples = np.asarray(tare_samples, dtype=float)
    env_count = len(masses_g)
    if env_count == 0 or any(mass <= 0 for mass in masses_g):
        raise ValueError("masses_g must contain positive values")
    if list(masses_g) != sorted(set(masses_g)):
        raise ValueError("masses_g must be strictly increasing and unique")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    if samples.ndim != 3 or samples.shape[1:] != (env_count, 6):
        raise ValueError(f"samples must have shape (S, {env_count}, 6), got {samples.shape}")
    if tare_samples.ndim != 3 or tare_samples.shape[1:] != (env_count, 6):
        raise ValueError(f"tare_samples must have shape (T, {env_count}, 6), got {tare_samples.shape}")
    if samples.shape[0] < 2 or tare_samples.shape[0] < 2:
        raise ValueError("At least two payload and tare samples are required")
    if not np.isfinite(samples).all() or not np.isfinite(tare_samples).all():
        raise ValueError("Wrench samples contain non-finite values")
    auxiliaries = {
        "settled": settled,
        "sample_max_relative_speed_m_s": sample_max_relative_speed_m_s,
        "sample_max_relative_angular_speed_rad_s": sample_max_relative_angular_speed_rad_s,
        "tool_normal_z": tool_normal_z,
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
        expected_fz = mass_g * 1.0e-3 * gravity_m_s2
        row: dict[str, float | int | bool] = {
            "env_id": env_id,
            "mass_g": int(mass_g),
            "mass_kg": mass_g * 1.0e-3,
            "expected_Fz_N": expected_fz,
            "settled": bool(settled[env_id]),
            "settle_time_s": float(settle_time_s),
            "sample_count": int(samples.shape[0]),
            "tare_sample_count": int(tare_samples.shape[0]),
            "sample_max_relative_speed_m_s": float(sample_max_relative_speed_m_s[env_id]),
            "sample_max_relative_angular_speed_rad_s": float(
                sample_max_relative_angular_speed_rad_s[env_id]
            ),
            "tool_normal_z": float(tool_normal_z[env_id]),
        }
        for index, (name, unit) in enumerate(zip(WRENCH_COMPONENTS, WRENCH_UNITS)):
            row[f"{name}_{unit}"] = float(corrected_mean[env_id, index])
            row[f"{name}_std_{unit}"] = float(corrected_std[env_id, index])
            row[f"raw_{name}_{unit}"] = float(raw_mean[env_id, index])
            row[f"raw_{name}_std_{unit}"] = float(raw_std[env_id, index])
            row[f"tare_{name}_{unit}"] = float(tare_mean[env_id, index])
            row[f"tare_{name}_std_{unit}"] = float(tare_std[env_id, index])
        row["Fz_error_N"] = float(corrected_mean[env_id, 2] - expected_fz)
        row["Fz_error_percent"] = float(
            100.0 * (corrected_mean[env_id, 2] - expected_fz) / expected_fz
        )
        rows.append(row)
    return rows


def regression_metrics(
    rows: Sequence[dict[str, object]],
    *,
    expected_tare_fz_n: float | None = None,
) -> dict[str, float | bool]:
    """Compute the normal-force linearity and centered-load cross-axis checks."""
    if len(rows) < 2:
        raise ValueError("At least two rows are required for regression")
    mass_kg = np.asarray([float(row["mass_kg"]) for row in rows])
    fz = np.asarray([float(row["Fz_N"]) for row in rows])
    slope, intercept = np.polyfit(mass_kg, fz, 1)
    fitted = slope * mass_kg + intercept
    residual = fz - fitted
    denominator = float(np.sum((fz - fz.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else 0.0
    expected = np.asarray([float(row["expected_Fz_N"]) for row in rows])
    rmse = float(np.sqrt(np.mean((fz - expected) ** 2)))
    max_abs_error = float(np.max(np.abs(fz - expected)))
    max_cross_force = max(
        max(abs(float(row["Fx_N"])), abs(float(row["Fy_N"]))) for row in rows
    )
    max_moment = max(
        max(abs(float(row[f"{name}_Nm"])) for name in ("Mx", "My", "Mz")) for row in rows
    )
    metrics: dict[str, float | bool] = {
        "slope_N_per_kg": float(slope),
        "intercept_N": float(intercept),
        "r_squared": float(r_squared),
        "fz_rmse_N": rmse,
        "max_abs_fz_error_N": max_abs_error,
        "fz_monotonic": bool(np.all(np.diff(fz) > 0.0)),
        "max_cross_axis_force_N": float(max_cross_force),
        "max_abs_moment_Nm": float(max_moment),
        "all_settled": all(bool(row["settled"]) for row in rows),
        "min_tool_normal_z": min(float(row["tool_normal_z"]) for row in rows),
    }
    if expected_tare_fz_n is not None:
        if not math.isfinite(expected_tare_fz_n):
            raise ValueError("expected_tare_fz_n must be finite")
        tare_fz = np.asarray([float(row["tare_Fz_N"]) for row in rows])
        metrics["mean_tare_fz_N"] = float(tare_fz.mean())
        metrics["max_abs_tare_fz_error_N"] = float(np.max(np.abs(tare_fz - expected_tare_fz_n)))
        metrics["tare_fz_span_N"] = float(np.ptp(tare_fz))
    return metrics


def evaluate_acceptance(
    metrics: dict[str, float | bool],
    *,
    gravity_m_s2: float,
    slope_relative_tolerance: float = 0.05,
    min_r_squared: float = 0.995,
    max_abs_intercept_n: float = 0.02,
    max_fz_rmse_n: float = 0.01,
    max_abs_fz_error_n: float = 0.02,
    max_abs_tare_fz_error_n: float = 0.02,
    max_cross_axis_force_n: float = 0.10,
    max_moment_nm: float = 0.01,
    min_tool_normal_z: float = 0.999,
) -> tuple[bool, list[str]]:
    """Return a conservative feasibility verdict and human-readable failures."""
    failures: list[str] = []
    numeric_metrics = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    ]
    if not all(math.isfinite(value) for value in numeric_metrics):
        failures.append("one or more acceptance metrics are non-finite")
    slope = float(metrics["slope_N_per_kg"])
    if abs(slope - gravity_m_s2) > slope_relative_tolerance * gravity_m_s2:
        failures.append(
            f"Fz slope {slope:.6f} N/kg is outside {slope_relative_tolerance:.1%} of g={gravity_m_s2:.6f}"
        )
    if float(metrics["r_squared"]) < min_r_squared:
        failures.append(
            f"Fz R^2 {float(metrics['r_squared']):.6f} is below {min_r_squared:.6f}"
        )
    if abs(float(metrics["intercept_N"])) > max_abs_intercept_n:
        failures.append(
            f"|Fz intercept| {abs(float(metrics['intercept_N'])):.6f} N exceeds {max_abs_intercept_n:.6f} N"
        )
    if float(metrics["fz_rmse_N"]) > max_fz_rmse_n:
        failures.append(
            f"Fz RMSE {float(metrics['fz_rmse_N']):.6f} N exceeds {max_fz_rmse_n:.6f} N"
        )
    if float(metrics["max_abs_fz_error_N"]) > max_abs_fz_error_n:
        failures.append(
            f"maximum |Fz-m*g| {float(metrics['max_abs_fz_error_N']):.6f} N exceeds "
            f"{max_abs_fz_error_n:.6f} N"
        )
    if not bool(metrics["fz_monotonic"]):
        failures.append("tare-corrected Fz is not strictly increasing with payload mass")
    if (
        "max_abs_tare_fz_error_N" in metrics
        and float(metrics["max_abs_tare_fz_error_N"]) > max_abs_tare_fz_error_n
    ):
        failures.append(
            f"tool-only tare Fz error {float(metrics['max_abs_tare_fz_error_N']):.6f} N exceeds "
            f"{max_abs_tare_fz_error_n:.6f} N"
        )
    if float(metrics["max_cross_axis_force_N"]) > max_cross_axis_force_n:
        failures.append(
            f"cross-axis force {float(metrics['max_cross_axis_force_N']):.6f} N exceeds {max_cross_axis_force_n:.6f} N"
        )
    if float(metrics["max_abs_moment_Nm"]) > max_moment_nm:
        failures.append(
            f"moment {float(metrics['max_abs_moment_Nm']):.6f} N*m exceeds {max_moment_nm:.6f} N*m"
        )
    if not bool(metrics["all_settled"]):
        failures.append("one or more payloads did not satisfy the stability window")
    if float(metrics["min_tool_normal_z"]) < min_tool_normal_z:
        failures.append(
            f"tool +Z alignment {float(metrics['min_tool_normal_z']):.6f} is below {min_tool_normal_z:.6f}"
        )
    return not failures, failures


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_csv(
    path: Path,
    masses_g: Sequence[int],
    samples: np.ndarray,
    tare_mean: np.ndarray,
    *,
    dt_s: float,
) -> None:
    """Write every payload-window sample, both raw and tare-corrected."""
    samples = np.asarray(samples, dtype=float)
    tare_mean = np.asarray(tare_mean, dtype=float)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if len(masses_g) == 0 or list(masses_g) != sorted(set(masses_g)):
        raise ValueError("masses_g must be strictly increasing and unique")
    if samples.shape[1:] != (len(masses_g), 6) or tare_mean.shape != (len(masses_g), 6):
        raise ValueError("Sample or tare shape does not match the mass sequence")
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError("At least two payload samples with shape (S, E, 6) are required")
    if not np.isfinite(samples).all() or not np.isfinite(tare_mean).all():
        raise ValueError("Sample or tare data contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    corrected_names = [f"{name}_{unit}" for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS)]
    raw_names = [f"raw_{name}_{unit}" for name, unit in zip(WRENCH_COMPONENTS, WRENCH_UNITS)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "time_s", "env_id", "mass_g", *corrected_names, *raw_names])
        for sample_index, sample in enumerate(samples):
            corrected = sample - tare_mean
            for env_id, mass_g in enumerate(masses_g):
                writer.writerow(
                    [
                        sample_index,
                        sample_index * dt_s,
                        env_id,
                        mass_g,
                        *corrected[env_id].tolist(),
                        *sample[env_id].tolist(),
                    ]
                )


def read_summary_csv(path: Path) -> list[dict[str, float | int | bool]]:
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
    masses_g = [int(row["mass_g"]) for row in rows]
    if masses_g != sorted(set(masses_g)):
        raise ValueError("Summary masses must be strictly increasing and unique")
    return rows
