"""Contracts for the added wrist_3 +10-degree cantilever experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "axia80_feasibility"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from axia80_feasibility.cantilever_model import (  # noqa: E402
    CANTILEVER_ARM_POSE,
    CANTILEVER_TOOL_SIZE_M,
)
from axia80_feasibility.measurement import write_summary_csv  # noqa: E402
from axia80_feasibility.model import PAYLOAD_SIZE_M, SENSOR_LINK_NAME  # noqa: E402
from axia80_feasibility.tilted_cantilever_measurement import (  # noqa: E402
    evaluate_tilted_acceptance,
    expected_payload_wrench_per_kg,
    expected_tool_tare_wrench,
    summarize_tilted_wrenches,
    tilted_metrics,
)
from axia80_feasibility.tilted_cantilever_model import (  # noqa: E402
    TILTED_ARM_POSE,
    TILTED_DYNAMIC_FRICTION,
    TILTED_STATIC_FRICTION,
    WRIST3_TILT_DEG,
    WRIST3_TILT_RAD,
    build_tilted_cantilever_model,
)
from axia80_feasibility.tilted_cantilever_plotting import (  # noqa: E402
    plot_tilted_cantilever_summary,
)
from test_axia80_feasibility import _link_transform  # noqa: E402


GRAVITY = 9.81


def _synthetic_rows() -> tuple[
    list[int],
    list[dict[str, float | int | bool]],
    np.ndarray,
    np.ndarray,
]:
    masses_g = [10, 50, 100, 250, 500]
    mass_kg = np.asarray(masses_g, dtype=float) * 1.0e-3
    lever = 0.1337
    normal_offset = 0.5 * CANTILEVER_TOOL_SIZE_M[0] + 0.5 * PAYLOAD_SIZE_M
    per_kg = expected_payload_wrench_per_kg(
        gravity_m_s2=GRAVITY,
        tilt_rad=WRIST3_TILT_RAD,
        bending_lever_arm_m=lever,
        payload_normal_offset_m=normal_offset,
    )
    expected_tare = expected_tool_tare_wrench(
        tool_mass_kg=0.20,
        gravity_m_s2=GRAVITY,
        tilt_rad=WRIST3_TILT_RAD,
        tool_center_z_m=lever,
    )
    env_count = len(masses_g)
    tare_center = np.tile(expected_tare, (env_count, 1))
    tare_center += np.arange(env_count)[:, None] * 1.0e-7
    tare_noise = np.asarray((2e-4, 2e-4, 1e-5, 2e-5, 2e-5, 1e-5))
    tare_samples = (
        tare_center[None, :, :]
        + np.asarray((-1.0, 0.0, 1.0))[:, None, None] * tare_noise
    )
    response = mass_kg[:, None] * per_kg[None, :]
    sample_noise = np.asarray((2e-4, 2e-4, 1e-5, 2e-5, 2e-5, 1e-5))
    samples = (
        tare_center[None, :, :]
        + response[None, :, :]
        + np.asarray((-1.5, -0.5, 0.5, 1.5))[:, None, None] * sample_noise
    )
    rows = summarize_tilted_wrenches(
        masses_g,
        samples,
        tare_samples,
        gravity_m_s2=GRAVITY,
        tilt_rad=WRIST3_TILT_RAD,
        bending_lever_arm_m=lever,
        payload_normal_offset_m=normal_offset,
        settled=[True] * env_count,
        settle_time_s=1.5,
        sample_max_relative_speed_m_s=[1e-3] * env_count,
        sample_max_relative_angular_speed_rad_s=[1e-2] * env_count,
        sample_max_position_error_m=[5e-4] * env_count,
        sample_max_abs_downhill_displacement_m=[2e-4] * env_count,
        sample_max_payload_tool_angle_deg=[0.1] * env_count,
        sample_min_edge_margin_m=[0.0398] * env_count,
        sample_min_broad_normal_z=[math.cos(WRIST3_TILT_RAD)] * env_count,
        sample_max_broad_normal_z=[math.cos(WRIST3_TILT_RAD)] * env_count,
        sample_max_abs_sensor_axis_z=[0.0] * env_count,
    )
    return masses_g, rows, per_kg, expected_tare


class TiltedGeometryTests(unittest.TestCase):
    def test_narrow_tool_retains_payload_margin(self) -> None:
        self.assertEqual(tuple(CANTILEVER_TOOL_SIZE_M), (0.012, 0.12, 0.24))
        lateral_margin = 0.5 * CANTILEVER_TOOL_SIZE_M[1] - 0.5 * PAYLOAD_SIZE_M
        self.assertAlmostEqual(lateral_margin, 0.04)

    def test_only_wrist3_changes_and_fk_matches_ten_degree_tilt(self) -> None:
        for name, value in CANTILEVER_ARM_POSE.items():
            expected = WRIST3_TILT_RAD if name == "wrist_3_joint" else value
            self.assertAlmostEqual(TILTED_ARM_POSE[name], expected)

        with tempfile.TemporaryDirectory(prefix="axia80-tilted-model-") as directory:
            urdf_path, metadata = build_tilted_cantilever_model(ROOT, Path(directory))
            robot = ET.parse(urdf_path).getroot()
        transform = _link_transform(robot, SENSOR_LINK_NAME, TILTED_ARM_POSE)
        world_x = transform[:3, :3] @ np.asarray((1.0, 0.0, 0.0))
        world_y = transform[:3, :3] @ np.asarray((0.0, 1.0, 0.0))
        world_z = transform[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
        self.assertAlmostEqual(float(world_x[2]), math.cos(WRIST3_TILT_RAD), places=8)
        self.assertAlmostEqual(float(world_y[2]), -math.sin(WRIST3_TILT_RAD), places=8)
        self.assertAlmostEqual(float(world_z[2]), 0.0, places=8)
        self.assertEqual(metadata["wrist3_tilt_deg"], 10.0)
        self.assertEqual(metadata["tool_size_sensor_frame_m"], (0.012, 0.12, 0.24))

    def test_friction_has_large_no_slide_margin(self) -> None:
        self.assertGreater(TILTED_STATIC_FRICTION, TILTED_DYNAMIC_FRICTION)
        self.assertGreater(TILTED_DYNAMIC_FRICTION, math.tan(WRIST3_TILT_RAD))
        self.assertGreater(TILTED_STATIC_FRICTION / math.tan(WRIST3_TILT_RAD), 10.0)


class TiltedMeasurementTests(unittest.TestCase):
    def test_synthetic_six_axis_distribution_passes(self) -> None:
        masses_g, rows, per_kg, expected_tare = _synthetic_rows()
        metrics = tilted_metrics(rows, expected_tare=expected_tare)
        self.assertEqual([row["mass_g"] for row in rows], masses_g)
        for index, (name, unit) in enumerate(
            zip(("fx", "fy", "fz", "mx", "my", "mz"), ("N", "N", "N", "Nm", "Nm", "Nm"))
        ):
            self.assertAlmostEqual(
                float(metrics[f"{name}_slope_{unit}_per_kg"]),
                float(per_kg[index]),
                places=11,
            )
        self.assertAlmostEqual(float(metrics["force_distribution_tilt_deg"]), 10.0, places=10)
        self.assertAlmostEqual(float(metrics["bending_distribution_tilt_deg"]), 10.0, places=10)
        passed, failures = evaluate_tilted_acceptance(
            metrics,
            expected_slopes=per_kg,
            position_tolerance_m=0.004,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_missing_lateral_distribution_fails(self) -> None:
        _, rows, per_kg, expected_tare = _synthetic_rows()
        for row in rows:
            row["Fy_N"] = 0.0
            row["Mx_Nm"] = 0.0
        metrics = tilted_metrics(rows, expected_tare=expected_tare)
        passed, failures = evaluate_tilted_acceptance(
            metrics,
            expected_slopes=per_kg,
            position_tolerance_m=0.004,
        )
        self.assertFalse(passed)
        self.assertTrue(any("Fy slope" in failure for failure in failures))
        self.assertTrue(any("Mx slope" in failure for failure in failures))

    def test_tilted_summary_plot_is_nonempty(self) -> None:
        _, rows, _, _ = _synthetic_rows()
        with tempfile.TemporaryDirectory(prefix="axia80-tilted-plot-") as directory:
            directory_path = Path(directory)
            csv_path = directory_path / "axia80_tilted_cantilever_summary.csv"
            png_path = directory_path / "axia80_tilted_cantilever_response.png"
            write_summary_csv(csv_path, rows)
            plot_tilted_cantilever_summary(csv_path, png_path, dpi=60)
            png = png_path.read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(png), 10_000)


if __name__ == "__main__":
    unittest.main()
