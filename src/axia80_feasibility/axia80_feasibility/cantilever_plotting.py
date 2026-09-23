"""Plot force transmission and bending response for the cantilever setup."""

from __future__ import annotations

from pathlib import Path

from .measurement import read_summary_csv


def plot_cantilever_summary(csv_path: Path, output_path: Path, *, dpi: int = 180) -> Path:
    """Create raw and tare-corrected six-axis plots with theoretical overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_summary_csv(csv_path)
    mass = np.asarray([float(row["mass_g"]) for row in rows])
    expected_fx = np.asarray([float(row["expected_Fx_N"]) for row in rows])
    expected_my = np.asarray([float(row["expected_My_Nm"]) for row in rows])
    expected_raw_fx = expected_fx + np.asarray(
        [float(row["tare_Fx_N"]) for row in rows]
    )
    expected_raw_my = expected_my + np.asarray(
        [float(row["tare_My_Nm"]) for row in rows]
    )
    force_colors = {"Fx": "tab:red", "Fy": "tab:green", "Fz": "tab:blue"}
    moment_colors = {"Mx": "tab:orange", "My": "tab:purple", "Mz": "tab:brown"}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True)
    for name, color in force_colors.items():
        axes[0, 0].plot(
            mass,
            [float(row[f"raw_{name}_N"]) for row in rows],
            label=name,
            color=color,
        )
        axes[1, 0].plot(
            mass,
            [float(row[f"{name}_N"]) for row in rows],
            label=name,
            color=color,
        )
    axes[0, 0].plot(
        mass,
        expected_raw_fx,
        "k--",
        linewidth=1.4,
        label="tare + expected Fx",
    )
    axes[1, 0].plot(mass, expected_fx, "k--", linewidth=1.4, label="expected m·g")

    for name, color in moment_colors.items():
        axes[0, 1].plot(
            mass,
            [float(row[f"raw_{name}_Nm"]) for row in rows],
            label=name,
            color=color,
        )
        axes[1, 1].plot(
            mass,
            [float(row[f"{name}_Nm"]) for row in rows],
            label=name,
            color=color,
        )
    axes[0, 1].plot(
        mass,
        expected_raw_my,
        "k--",
        linewidth=1.4,
        label="tare + expected My",
    )
    axes[1, 1].plot(
        mass,
        expected_my,
        "k--",
        linewidth=1.4,
        label="expected m·g·d",
    )

    axes[0, 0].set_title("Raw reaction force (tool weight included)")
    axes[0, 1].set_title("Raw bending reaction (tool weight included)")
    axes[1, 0].set_title("Tare-corrected payload force")
    axes[1, 1].set_title("Tare-corrected payload bending moment")
    axes[0, 0].set_ylabel("Force [N]")
    axes[1, 0].set_ylabel("Force [N]")
    axes[0, 1].set_ylabel("Moment [N·m]")
    axes[1, 1].set_ylabel("Moment [N·m]")
    axes[1, 0].set_xlabel("Payload mass [g]")
    axes[1, 1].set_xlabel("Payload mass [g]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.28)
        axis.legend(ncol=2)
        axis.axhline(0.0, color="0.4", linewidth=0.7)
    fig.suptitle(
        "UR5e – Axia80 – thin-face cantilever: force and bending response",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

