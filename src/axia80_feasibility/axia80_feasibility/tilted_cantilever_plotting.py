"""Visualize six-axis load distribution for the tilted cantilever test."""

from __future__ import annotations

from pathlib import Path

from .measurement import read_summary_csv


def plot_tilted_cantilever_summary(
    csv_path: Path,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot measured and theoretical raw/tare-corrected wrench components."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_summary_csv(csv_path)
    mass = np.asarray([float(row["mass_g"]) for row in rows])
    tilt_deg = float(rows[0]["tilt_deg"])
    force_colors = {"Fx": "tab:red", "Fy": "tab:green", "Fz": "tab:blue"}
    moment_colors = {"Mx": "tab:orange", "My": "tab:purple", "Mz": "tab:brown"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    for name, color in force_colors.items():
        raw = np.asarray([float(row[f"raw_{name}_N"]) for row in rows])
        corrected = np.asarray([float(row[f"{name}_N"]) for row in rows])
        expected = np.asarray([float(row[f"expected_{name}_N"]) for row in rows])
        tare = np.asarray([float(row[f"tare_{name}_N"]) for row in rows])
        axes[0, 0].plot(mass, raw, color=color, label=name)
        axes[0, 0].plot(
            mass,
            tare + expected,
            "--",
            color=color,
            linewidth=1.2,
            label=f"expected {name}",
        )
        axes[1, 0].plot(mass, corrected, color=color, label=name)
        axes[1, 0].plot(
            mass,
            expected,
            "--",
            color=color,
            linewidth=1.2,
            label=f"expected {name}",
        )

    for name, color in moment_colors.items():
        raw = np.asarray([float(row[f"raw_{name}_Nm"]) for row in rows])
        corrected = np.asarray([float(row[f"{name}_Nm"]) for row in rows])
        expected = np.asarray([float(row[f"expected_{name}_Nm"]) for row in rows])
        tare = np.asarray([float(row[f"tare_{name}_Nm"]) for row in rows])
        axes[0, 1].plot(mass, raw, color=color, label=name)
        axes[0, 1].plot(
            mass,
            tare + expected,
            "--",
            color=color,
            linewidth=1.2,
            label=f"expected {name}",
        )
        axes[1, 1].plot(mass, corrected, color=color, label=name)
        axes[1, 1].plot(
            mass,
            expected,
            "--",
            color=color,
            linewidth=1.2,
            label=f"expected {name}",
        )

    axes[0, 0].set_title("Raw distributed force (tool weight included)")
    axes[0, 1].set_title("Raw distributed moment (tool weight included)")
    axes[1, 0].set_title("Tare-corrected distributed payload force")
    axes[1, 1].set_title("Tare-corrected distributed payload moment")
    axes[0, 0].set_ylabel("Force [N]")
    axes[1, 0].set_ylabel("Force [N]")
    axes[0, 1].set_ylabel("Moment [N·m]")
    axes[1, 1].set_ylabel("Moment [N·m]")
    axes[1, 0].set_xlabel("Payload mass [g]")
    axes[1, 1].set_xlabel("Payload mass [g]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.28)
        axis.legend(ncol=2, fontsize=8)
        axis.axhline(0.0, color="0.4", linewidth=0.7)
    fig.suptitle(
        f"UR5e – Axia80 – wrist_3 +{tilt_deg:g}°: six-axis load distribution",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path

