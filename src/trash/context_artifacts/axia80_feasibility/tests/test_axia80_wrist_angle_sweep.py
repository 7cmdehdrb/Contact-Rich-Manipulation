"""Pure-Python contracts for the fixed-mass wrist Roll x Tilt sweep."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT
WORKSPACE_ROOT = Path(os.environ.get("AXIA80_WORKSPACE_ROOT", ROOT)).resolve()
if "AXIA80_WORKSPACE_ROOT" not in os.environ:
    for candidate in ROOT.parents:
        if (candidate / "IsaacLab").is_dir() and (
            candidate / "src/Universal_Robots_ROS2_Description"
        ).is_dir():
            WORKSPACE_ROOT = candidate
            break
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from axia80_feasibility.measurement import write_summary_csv  # noqa: E402
from axia80_feasibility.model import SENSOR_LINK_NAME  # noqa: E402
from axia80_feasibility.wrist_angle_measurement import (  # noqa: E402
    evaluate_wrist_angle_acceptance,
    nominal_payload_wrench,
    nominal_tool_tare_wrench,
    read_wrist_angle_summary_csv,
    summarize_wrist_angle_wrenches,
    write_wrist_angle_sample_csv,
    wrist_angle_metrics,
)
from axia80_feasibility.wrist_angle_model import (  # noqa: E402
    DEFAULT_ROLL_ANGLES_DEG,
    DEFAULT_TILT_ANGLES_DEG,
    angle_grid,
    arm_pose_for_angles,
    build_wrist_angle_model,
)
from axia80_feasibility.wrist_angle_plotting import (  # noqa: E402
    plot_wrist_angle_center_slices,
    plot_wrist_angle_error_heatmaps,
    plot_wrist_angle_fixed_roll_responses,
    plot_wrist_angle_fixed_tilt_responses,
    plot_wrist_angle_heatmaps,
)
from test_axia80_feasibility import _link_transform  # noqa: E402


GRAVITY = 9.81
PAYLOAD_MASS_G = 500
LEVER_M = 0.1337
NORMAL_OFFSET_M = 0.026
TOOL_MASS_KG = 0.20


def _synthetic_measurement():
    pairs = angle_grid(DEFAULT_ROLL_ANGLES_DEG, DEFAULT_TILT_ANGLES_DEG)
    env_count = len(pairs)
    response = np.asarray(
        [
            nominal_payload_wrench(
                mass_kg=PAYLOAD_MASS_G * 1.0e-3,
                gravity_m_s2=GRAVITY,
                roll_deg=roll,
                tilt_deg=tilt,
                bending_lever_arm_m=LEVER_M,
                payload_normal_offset_m=NORMAL_OFFSET_M,
            )
            for roll, tilt in pairs
        ]
    )
    tare = np.asarray(
        [
            nominal_tool_tare_wrench(
                tool_mass_kg=TOOL_MASS_KG,
                gravity_m_s2=GRAVITY,
                roll_deg=roll,
                tilt_deg=tilt,
                tool_center_z_m=LEVER_M,
            )
            for roll, tilt in pairs
        ]
    )
    tare_noise = np.asarray((1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5))
    tare_samples = tare[None, :, :] + np.asarray((-1.0, 0.0, 1.0))[:, None, None] * tare_noise
    sample_noise = np.asarray((2e-4, 2e-4, 2e-4, 2e-5, 2e-5, 2e-5))
    samples = (
        tare[None, :, :]
        + response[None, :, :]
        + np.asarray((-1.5, -0.5, 0.5, 1.5))[:, None, None] * sample_noise
    )
    expected_samples = np.repeat(response[None, :, :], samples.shape[0], axis=0)
    expected_tare_samples = np.repeat(tare[None, :, :], tare_samples.shape[0], axis=0)
    actual_roll = [roll for roll, _ in pairs]
    actual_tilt = [tilt for _, tilt in pairs]
    inclination = [
        math.degrees(
            math.acos(math.cos(math.radians(roll)) * math.cos(math.radians(tilt)))
        )
        for roll, tilt in pairs
    ]
    required_mu = [math.tan(math.radians(value)) for value in inclination]
    rows = summarize_wrist_angle_wrenches(
        pairs,
        PAYLOAD_MASS_G,
        samples,
        tare_samples,
        expected_samples,
        expected_tare_samples,
        gravity_m_s2=GRAVITY,
        bending_lever_arm_m=LEVER_M,
        payload_normal_offset_m=NORMAL_OFFSET_M,
        settled=[True] * env_count,
        settle_time_s=1.5,
        sample_mean_actual_roll_deg=actual_roll,
        sample_mean_actual_tilt_deg=actual_tilt,
        sample_max_combined_inclination_deg=inclination,
        sample_max_required_static_friction=required_mu,
        sample_max_relative_speed_m_s=[1e-3] * env_count,
        sample_max_relative_angular_speed_rad_s=[1e-2] * env_count,
        sample_max_position_error_m=[5e-4] * env_count,
        sample_max_tangent_displacement_m=[3e-4] * env_count,
        sample_max_payload_tool_angle_deg=[0.1] * env_count,
        sample_min_width_edge_margin_m=[0.0397] * env_count,
        sample_min_length_edge_margin_m=[0.0997] * env_count,
    )
    return pairs, samples, tare_samples, expected_samples, rows


class WristAngleGeometryTests(unittest.TestCase):
    def test_grid_is_tilt_major_and_validated(self) -> None:
        grid = angle_grid((-10.0, 0.0, 10.0), (-5.0, 5.0))
        self.assertEqual(
            grid,
            [(-10.0, -5.0), (0.0, -5.0), (10.0, -5.0), (-10.0, 5.0), (0.0, 5.0), (10.0, 5.0)],
        )
        for rolls, tilts in (((0.0, 0.0), (-5.0, 5.0)), ((31.0,), (0.0,))):
            with self.subTest(rolls=rolls, tilts=tilts), self.assertRaises(ValueError):
                angle_grid(rolls, tilts)

    def test_wrist_joint_offsets_match_fk_force_convention(self) -> None:
        roll_deg, tilt_deg = 10.0, -10.0
        with tempfile.TemporaryDirectory(prefix="axia80-wrist-angle-model-") as directory:
            urdf_path, metadata = build_wrist_angle_model(
                WORKSPACE_ROOT, Path(directory)
            )
            robot = ET.parse(urdf_path).getroot()
        transform = _link_transform(
            robot,
            SENSOR_LINK_NAME,
            arm_pose_for_angles(roll_deg, tilt_deg),
        )
        world_up_sensor = transform[:3, :3].T @ np.asarray((0.0, 0.0, 1.0))
        roll = math.radians(roll_deg)
        tilt = math.radians(tilt_deg)
        expected = np.asarray(
            (
                math.cos(roll) * math.cos(tilt),
                -math.sin(roll) * math.cos(tilt),
                -math.sin(tilt),
            )
        )
        np.testing.assert_allclose(world_up_sensor, expected, atol=1e-8)
        self.assertEqual(metadata["roll_joint_name"], "wrist_3_joint")
        self.assertEqual(metadata["tilt_joint_name"], "wrist_2_joint")
        self.assertEqual(metadata["tool_size_sensor_frame_m"], (0.012, 0.12, 0.24))

    def test_nominal_wrench_obeys_force_and_moment_identities(self) -> None:
        wrench = nominal_payload_wrench(
            mass_kg=0.5,
            gravity_m_s2=GRAVITY,
            roll_deg=10.0,
            tilt_deg=-5.0,
            bending_lever_arm_m=LEVER_M,
            payload_normal_offset_m=NORMAL_OFFSET_M,
        )
        self.assertAlmostEqual(float(np.linalg.norm(wrench[:3])), 0.5 * GRAVITY, places=12)
        self.assertAlmostEqual(wrench[3], -LEVER_M * wrench[1], places=12)
        self.assertAlmostEqual(
            wrench[4], LEVER_M * wrench[0] - NORMAL_OFFSET_M * wrench[2], places=12
        )
        self.assertAlmostEqual(wrench[5], NORMAL_OFFSET_M * wrench[1], places=12)


class WristAngleMeasurementTests(unittest.TestCase):
    def test_synthetic_grid_passes_pointwise_acceptance(self) -> None:
        pairs, _, _, _, rows = _synthetic_measurement()
        metrics = wrist_angle_metrics(rows)
        self.assertEqual(len(rows), len(pairs))
        self.assertTrue(metrics["complete_angle_grid"])
        self.assertAlmostEqual(float(metrics["force_component_rmse_N"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["moment_component_rmse_Nm"]), 0.0, places=12)
        passed, failures = evaluate_wrist_angle_acceptance(
            metrics,
            position_tolerance_m=0.006,
            orientation_tolerance_deg=2.0,
            static_friction=2.0,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_wrong_force_distribution_fails(self) -> None:
        _, _, _, _, rows = _synthetic_measurement()
        for row in rows:
            row["Fy_N"] = 0.0
            row["Fy_error_N"] = -float(row["expected_Fy_N"])
            force_error = np.asarray(
                [float(row[f"{name}_error_N"]) for name in ("Fx", "Fy", "Fz")]
            )
            row["force_vector_error_N"] = float(np.linalg.norm(force_error))
        metrics = wrist_angle_metrics(rows)
        passed, failures = evaluate_wrist_angle_acceptance(
            metrics,
            position_tolerance_m=0.006,
            orientation_tolerance_deg=2.0,
            static_friction=2.0,
        )
        self.assertFalse(passed)
        self.assertTrue(any("force component" in failure for failure in failures))

    def test_csv_and_all_three_plots_are_nonempty(self) -> None:
        pairs, samples, tare_samples, expected_samples, rows = _synthetic_measurement()
        with tempfile.TemporaryDirectory(prefix="axia80-wrist-angle-plots-") as directory:
            directory_path = Path(directory)
            summary = directory_path / "summary.csv"
            sample_csv = directory_path / "samples.csv"
            response = directory_path / "response.png"
            error = directory_path / "error.png"
            slices = directory_path / "slices.png"
            fixed_roll = directory_path / "fixed_roll.png"
            fixed_tilt = directory_path / "fixed_tilt.png"
            write_summary_csv(summary, rows)
            parsed = read_wrist_angle_summary_csv(summary)
            self.assertEqual(len(parsed), len(pairs))
            write_wrist_angle_sample_csv(
                sample_csv,
                pairs,
                PAYLOAD_MASS_G,
                samples,
                tare_samples.mean(axis=0),
                expected_samples,
                dt_s=1.0 / 240.0,
            )
            plot_wrist_angle_heatmaps(summary, response, dpi=60)
            plot_wrist_angle_error_heatmaps(summary, error, dpi=60)
            plot_wrist_angle_center_slices(summary, slices, dpi=60)
            plot_wrist_angle_fixed_roll_responses(summary, fixed_roll, dpi=60)
            plot_wrist_angle_fixed_tilt_responses(summary, fixed_tilt, dpi=60)
            self.assertEqual(len(sample_csv.read_text().splitlines()), 1 + len(pairs) * len(samples))
            for path in (response, error, slices, fixed_roll, fixed_tilt):
                data = path.read_bytes()
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(len(data), 10_000)


if __name__ == "__main__":
    unittest.main()
