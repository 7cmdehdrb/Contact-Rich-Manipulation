"""Pure-Python contracts for the added thin-face cantilever experiment."""

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

from axia80_feasibility.cantilever_measurement import (  # noqa: E402
    cantilever_metrics,
    evaluate_cantilever_acceptance,
    summarize_cantilever_wrenches,
)
from axia80_feasibility.cantilever_model import (  # noqa: E402
    CANTILEVER_ARM_POSE,
    CANTILEVER_TOOL_GAP_M,
    CANTILEVER_TOOL_MASS_KG,
    CANTILEVER_TOOL_SIZE_M,
    _append_axia80_and_cantilever_tool,
    build_cantilever_model,
)
from axia80_feasibility.cantilever_plotting import plot_cantilever_summary  # noqa: E402
from axia80_feasibility.measurement import write_summary_csv  # noqa: E402
from axia80_feasibility.model import (  # noqa: E402
    PAYLOAD_SIZE_M,
    SENSOR_HEIGHT_M,
    SENSOR_LINK_NAME,
    SENSOR_RADIUS_M,
    TOOL_LINK_NAME,
)
from test_axia80_feasibility import _link_transform, _numbers  # noqa: E402


GRAVITY = 9.81


def _synthetic_rows() -> tuple[list[int], list[dict[str, float | int | bool]]]:
    masses_g = [10, 50, 100, 250, 500]
    mass_kg = np.asarray(masses_g, dtype=float) * 1.0e-3
    env_count = len(masses_g)
    lever = 0.5 * SENSOR_HEIGHT_M + CANTILEVER_TOOL_GAP_M + 0.12

    tare = np.zeros((env_count, 6), dtype=float)
    tare[:, 0] = CANTILEVER_TOOL_MASS_KG * GRAVITY
    tare[:, 4] = CANTILEVER_TOOL_MASS_KG * GRAVITY * lever
    tare += np.arange(env_count)[:, None] * 1.0e-6
    tare_noise = np.asarray((1.0e-4, 2.0e-5, 2.0e-5, 1.0e-5, 1.0e-5, 1.0e-5))
    tare_samples = tare[None, :, :] + np.asarray((-1.0, 0.0, 1.0))[:, None, None] * tare_noise

    response = np.zeros((env_count, 6), dtype=float)
    response[:, 0] = mass_kg * GRAVITY
    response[:, 4] = mass_kg * GRAVITY * lever
    sample_noise = np.asarray((2.0e-4, 2.0e-5, 2.0e-5, 1.0e-5, 2.0e-5, 1.0e-5))
    samples = (
        tare[None, :, :]
        + response[None, :, :]
        + np.asarray((-1.5, -0.5, 0.5, 1.5))[:, None, None] * sample_noise
    )
    rows = summarize_cantilever_wrenches(
        masses_g,
        samples,
        tare_samples,
        gravity_m_s2=GRAVITY,
        bending_lever_arm_m=lever,
        settled=[True] * env_count,
        settle_time_s=1.5,
        sample_max_relative_speed_m_s=[1.0e-3] * env_count,
        sample_max_relative_angular_speed_rad_s=[1.0e-2] * env_count,
        sample_max_position_error_m=[1.0e-3] * env_count,
        sample_min_broad_normal_z=[1.0] * env_count,
        sample_max_abs_sensor_axis_z=[0.0] * env_count,
    )
    return masses_g, rows


class CantileverGeometryTests(unittest.TestCase):
    def test_thin_end_face_is_mounted_beyond_the_axia_face(self) -> None:
        robot = ET.Element("robot", {"name": "cantilever_contract"})
        ET.SubElement(robot, "link", {"name": "tool0"})
        _append_axia80_and_cantilever_tool(robot)

        sensor = robot.find(f"link[@name='{SENSOR_LINK_NAME}']")
        tool = robot.find(f"link[@name='{TOOL_LINK_NAME}']")
        cylinder = sensor.find("collision/geometry/cylinder")
        self.assertAlmostEqual(float(cylinder.get("radius")), SENSOR_RADIUS_M)
        self.assertAlmostEqual(float(cylinder.get("length")), SENSOR_HEIGHT_M)

        box = tool.find("collision/geometry/box")
        size = _numbers(box, "size")
        np.testing.assert_allclose(size, (0.012, 0.12, 0.24))
        center = _numbers(tool.find("collision/origin"), "xyz")
        self.assertAlmostEqual(center[2] - 0.5 * size[2], 0.5 * SENSOR_HEIGHT_M + 0.001)
        self.assertEqual(tuple(CANTILEVER_TOOL_SIZE_M), (0.012, 0.12, 0.24))

        measurement = robot.find("joint[@name='axia80_cantilever_measurement_joint']")
        self.assertEqual(measurement.find("parent").get("link"), SENSOR_LINK_NAME)
        self.assertEqual(measurement.find("child").get("link"), TOOL_LINK_NAME)
        np.testing.assert_allclose(_numbers(measurement.find("origin"), "xyz"), 0.0)
        np.testing.assert_allclose(_numbers(measurement.find("origin"), "rpy"), 0.0)

    def test_generated_pose_makes_broad_normal_up_and_cantilever_horizontal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axia80-cantilever-model-") as directory:
            urdf_path, metadata = build_cantilever_model(ROOT, Path(directory))
            robot = ET.parse(urdf_path).getroot()

        sensor_transform = _link_transform(robot, SENSOR_LINK_NAME, CANTILEVER_ARM_POSE)
        tool_transform = _link_transform(robot, TOOL_LINK_NAME, CANTILEVER_ARM_POSE)
        np.testing.assert_allclose(sensor_transform, tool_transform, atol=1.0e-9)
        world_broad_normal = sensor_transform[:3, :3] @ np.asarray((1.0, 0.0, 0.0))
        world_cantilever_axis = sensor_transform[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
        np.testing.assert_allclose(world_broad_normal, (0.0, 0.0, 1.0), atol=1.0e-8)
        self.assertAlmostEqual(float(world_cantilever_axis[2]), 0.0, places=8)
        self.assertEqual(metadata["tool_mount_face_size_m"], [0.012, 0.12])
        self.assertEqual(metadata["tool_mount_face_outward_normal_axis"], "-Z")
        self.assertEqual(metadata["tool_extension_axis"], "+Z")
        self.assertEqual(metadata["tool_broad_face_size_m"], [0.12, 0.24])
        self.assertEqual(metadata["payload_center_in_sensor_m"][1], 0.0)
        self.assertAlmostEqual(
            float(metadata["payload_center_in_sensor_m"][0]),
            0.5 * 0.012 + 0.5 * PAYLOAD_SIZE_M,
        )
        self.assertAlmostEqual(
            float(metadata["payload_bending_lever_arm_m"]),
            0.5 * SENSOR_HEIGHT_M + CANTILEVER_TOOL_GAP_M + 0.12,
        )


class CantileverMeasurementTests(unittest.TestCase):
    def test_expected_fx_and_my_pass_regression_and_tare_checks(self) -> None:
        masses_g, rows = _synthetic_rows()
        lever = float(rows[0]["bending_lever_arm_m"])
        expected_tare_fx = CANTILEVER_TOOL_MASS_KG * GRAVITY
        expected_tare_my = expected_tare_fx * lever
        metrics = cantilever_metrics(
            rows,
            expected_tare_fx_n=expected_tare_fx,
            expected_tare_my_nm=expected_tare_my,
        )
        self.assertEqual([row["mass_g"] for row in rows], masses_g)
        self.assertAlmostEqual(float(metrics["fx_slope_N_per_kg"]), GRAVITY, places=12)
        self.assertAlmostEqual(
            float(metrics["my_slope_Nm_per_kg"]),
            GRAVITY * lever,
            places=12,
        )
        self.assertAlmostEqual(float(metrics["max_abs_lever_arm_error_m"]), 0.0, places=12)
        passed, failures = evaluate_cantilever_acceptance(
            metrics,
            gravity_m_s2=GRAVITY,
            bending_lever_arm_m=lever,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_missing_bending_response_fails_acceptance(self) -> None:
        _, rows = _synthetic_rows()
        lever = float(rows[0]["bending_lever_arm_m"])
        for row in rows:
            row["My_Nm"] = 1.0e-6 * float(row["mass_g"])
            row["measured_lever_arm_m"] = float(row["My_Nm"]) / float(row["Fx_N"])
        metrics = cantilever_metrics(
            rows,
            expected_tare_fx_n=CANTILEVER_TOOL_MASS_KG * GRAVITY,
            expected_tare_my_nm=CANTILEVER_TOOL_MASS_KG * GRAVITY * lever,
        )
        passed, failures = evaluate_cantilever_acceptance(
            metrics,
            gravity_m_s2=GRAVITY,
            bending_lever_arm_m=lever,
        )
        self.assertFalse(passed)
        self.assertTrue(any("My slope" in failure for failure in failures))
        self.assertTrue(any("lever arm" in failure for failure in failures))

    def test_summary_can_be_plotted(self) -> None:
        _, rows = _synthetic_rows()
        with tempfile.TemporaryDirectory(prefix="axia80-cantilever-plot-") as directory:
            directory_path = Path(directory)
            csv_path = directory_path / "axia80_cantilever_summary.csv"
            png_path = directory_path / "axia80_cantilever_response.png"
            write_summary_csv(csv_path, rows)
            plot_cantilever_summary(csv_path, png_path, dpi=60)
            png = png_path.read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(png), 10_000)


if __name__ == "__main__":
    unittest.main()
