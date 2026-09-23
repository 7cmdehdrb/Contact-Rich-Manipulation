"""Plots for the fixed-mass wrist roll/tilt response grid."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .measurement import WRENCH_COMPONENTS, WRENCH_UNITS
from .wrist_angle_measurement import read_wrist_angle_summary_csv


def _grid(
    rows: Sequence[dict[str, object]],
    column: str,
) -> tuple[object, object, object]:
    import numpy as np

    rolls = np.asarray(sorted({float(row["commanded_roll_deg"]) for row in rows}))
    tilts = np.asarray(sorted({float(row["commanded_tilt_deg"]) for row in rows}))
    values = np.full((len(tilts), len(rolls)), np.nan, dtype=float)
    roll_index = {value: index for index, value in enumerate(rolls)}
    tilt_index = {value: index for index, value in enumerate(tilts)}
    for row in rows:
        values[
            tilt_index[float(row["commanded_tilt_deg"])],
            roll_index[float(row["commanded_roll_deg"])],
        ] = float(row[column])
    if not np.isfinite(values).all():
        raise ValueError("Angle rows do not form a complete roll/tilt grid")
    return rolls, tilts, values


def _heatmap_figure(
    rows: Sequence[dict[str, object]],
    output_path: Path,
    *,
    error: bool,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, TwoSlopeNorm

    mass_g = int(rows[0]["mass_g"])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True, sharey=True)
    for axis, name, unit in zip(axes.flat, WRENCH_COMPONENTS, WRENCH_UNITS):
        column = f"{name}_error_{unit}" if error else f"{name}_{unit}"
        rolls, tilts, values = _grid(rows, column)
        low = float(values.min())
        high = float(values.max())
        if low < 0.0 < high:
            norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
        elif low == high:
            norm = Normalize(vmin=low - 1.0, vmax=high + 1.0)
        else:
            norm = Normalize(vmin=low, vmax=high)
        image = axis.pcolormesh(
            rolls,
            tilts,
            values,
            shading="nearest",
            cmap="coolwarm",
            norm=norm,
        )
        axis.set_title(f"{name} [{unit}]")
        axis.set_xlabel("Roll / wrist_3 offset [deg]")
        axis.set_ylabel("Tilt / wrist_2 offset [deg]")
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.45)
        axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.45)
        fig.colorbar(image, ax=axis, shrink=0.88)
        if len(rolls) <= 7 and len(tilts) <= 7:
            span = max(abs(low), abs(high), 1.0e-12)
            for tilt_index, tilt in enumerate(tilts):
                for roll_index, roll in enumerate(rolls):
                    value = float(values[tilt_index, roll_index])
                    color = "white" if abs(value) > 0.58 * span else "black"
                    axis.text(
                        roll,
                        tilt,
                        f"{value:.3g}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
    description = "measured minus actual-pose expected" if error else "tare-corrected measured"
    fig.suptitle(
        f"UR5e–Axia80 fixed {mass_g} g payload: {description} wrench",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_wrist_angle_heatmaps(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot measured Fx..Mz as six roll-by-tilt heatmaps."""
    return _heatmap_figure(
        read_wrist_angle_summary_csv(csv_path),
        output_path,
        error=False,
        dpi=dpi,
    )


def plot_wrist_angle_error_heatmaps(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot measured-minus-model residuals over the same angle grid."""
    return _heatmap_figure(
        read_wrist_angle_summary_csv(csv_path),
        output_path,
        error=True,
        dpi=dpi,
    )


def plot_wrist_angle_center_slices(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot roll at zero tilt and tilt at zero roll with model overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_wrist_angle_summary_csv(csv_path)
    if not any(abs(float(row["commanded_roll_deg"])) < 1.0e-12 for row in rows):
        raise ValueError("Center-slice plot requires a zero roll command")
    if not any(abs(float(row["commanded_tilt_deg"])) < 1.0e-12 for row in rows):
        raise ValueError("Center-slice plot requires a zero tilt command")
    roll_rows = sorted(
        (row for row in rows if abs(float(row["commanded_tilt_deg"])) < 1.0e-12),
        key=lambda row: float(row["commanded_roll_deg"]),
    )
    tilt_rows = sorted(
        (row for row in rows if abs(float(row["commanded_roll_deg"])) < 1.0e-12),
        key=lambda row: float(row["commanded_tilt_deg"]),
    )
    mass_g = int(rows[0]["mass_g"])
    force_colors = {"Fx": "tab:red", "Fy": "tab:green", "Fz": "tab:blue"}
    moment_colors = {"Mx": "tab:orange", "My": "tab:purple", "Mz": "tab:brown"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))

    def plot_slice(axis, selected_rows, angle_key, names, colors, unit, title):
        angles = np.asarray([float(row[angle_key]) for row in selected_rows])
        for name in names:
            measured = np.asarray([float(row[f"{name}_{unit}"]) for row in selected_rows])
            expected = np.asarray(
                [float(row[f"expected_{name}_{unit}"]) for row in selected_rows]
            )
            axis.plot(angles, measured, "o-", color=colors[name], label=name)
            axis.plot(angles, expected, "--", color=colors[name], linewidth=1.2)
        axis.set_title(title)
        axis.grid(True, alpha=0.28)
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.legend(ncol=3, fontsize=8)

    plot_slice(
        axes[0, 0],
        roll_rows,
        "commanded_roll_deg",
        ("Fx", "Fy", "Fz"),
        force_colors,
        "N",
        "Force vs Roll at Tilt=0°",
    )
    plot_slice(
        axes[0, 1],
        roll_rows,
        "commanded_roll_deg",
        ("Mx", "My", "Mz"),
        moment_colors,
        "Nm",
        "Moment vs Roll at Tilt=0°",
    )
    plot_slice(
        axes[1, 0],
        tilt_rows,
        "commanded_tilt_deg",
        ("Fx", "Fy", "Fz"),
        force_colors,
        "N",
        "Force vs Tilt at Roll=0°",
    )
    plot_slice(
        axes[1, 1],
        tilt_rows,
        "commanded_tilt_deg",
        ("Mx", "My", "Mz"),
        moment_colors,
        "Nm",
        "Moment vs Tilt at Roll=0°",
    )
    axes[0, 0].set_ylabel("Force [N]")
    axes[1, 0].set_ylabel("Force [N]")
    axes[0, 1].set_ylabel("Moment [N·m]")
    axes[1, 1].set_ylabel("Moment [N·m]")
    axes[0, 0].set_xlabel("Roll [deg]")
    axes[0, 1].set_xlabel("Roll [deg]")
    axes[1, 0].set_xlabel("Tilt [deg]")
    axes[1, 1].set_xlabel("Tilt [deg]")
    fig.suptitle(
        f"UR5e–Axia80 fixed {mass_g} g payload: measured (solid) vs expected (dashed)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_fixed_angle_responses(
    csv_path: Path,
    output_path: Path,
    *,
    fixed_key: str,
    varying_key: str,
    fixed_label: str,
    varying_label: str,
    dpi: int,
) -> Path:
    """Plot all six measured channels for several fixed-angle slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_wrist_angle_summary_csv(csv_path)
    fixed_values = sorted({float(row[fixed_key]) for row in rows})
    varying_values = sorted({float(row[varying_key]) for row in rows})
    if len(fixed_values) < 2 or len(varying_values) < 2:
        raise ValueError("Fixed-angle response plots require at least a 2x2 angle grid")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True)
    colors = plt.get_cmap("coolwarm")
    for axis, name, unit in zip(axes.flat, WRENCH_COMPONENTS, WRENCH_UNITS):
        for index, fixed_value in enumerate(fixed_values):
            selected = sorted(
                (
                    row
                    for row in rows
                    if abs(float(row[fixed_key]) - fixed_value) < 1.0e-12
                ),
                key=lambda row: float(row[varying_key]),
            )
            if [float(row[varying_key]) for row in selected] != varying_values:
                raise ValueError("Summary rows do not form a complete Roll/Tilt grid")
            color = colors(index / max(1, len(fixed_values) - 1))
            axis.plot(
                varying_values,
                [float(row[f"{name}_{unit}"]) for row in selected],
                "o-",
                color=color,
                linewidth=1.7,
                markersize=4.5,
                label=f"{fixed_label}={fixed_value:g}°",
            )
        axis.set_title(f"{name} [{unit}]")
        axis.set_xlabel(f"{varying_label} [deg]")
        axis.set_ylabel(unit.replace("Nm", "N·m"))
        axis.grid(True, alpha=0.28)
        axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.legend(fontsize=7.5, ncol=1)

    mass_g = int(rows[0]["mass_g"])
    fig.suptitle(
        f"UR5e–Axia80 fixed {mass_g} g: response vs {varying_label} "
        f"at five fixed {fixed_label} values",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_wrist_angle_fixed_roll_responses(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot response versus Tilt for each fixed Roll value in the CSV."""
    return _plot_fixed_angle_responses(
        csv_path,
        output_path,
        fixed_key="commanded_roll_deg",
        varying_key="commanded_tilt_deg",
        fixed_label="Roll",
        varying_label="Tilt",
        dpi=dpi,
    )


def plot_wrist_angle_fixed_tilt_responses(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot response versus Roll for each fixed Tilt value in the CSV."""
    return _plot_fixed_angle_responses(
        csv_path,
        output_path,
        fixed_key="commanded_tilt_deg",
        varying_key="commanded_roll_deg",
        fixed_label="Tilt",
        varying_label="Roll",
        dpi=dpi,
    )


__all__ = [
    "plot_wrist_angle_center_slices",
    "plot_wrist_angle_error_heatmaps",
    "plot_wrist_angle_fixed_roll_responses",
    "plot_wrist_angle_fixed_tilt_responses",
    "plot_wrist_angle_heatmaps",
]
