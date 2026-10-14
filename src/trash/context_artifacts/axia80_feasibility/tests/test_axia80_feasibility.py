"""Pure-Python contracts for the Axia80 static-payload feasibility package.

These tests intentionally do not import Isaac Lab or start Kit.  They exercise
the numerical reduction, artifacts, and generated URDF that the simulator
runner consumes.
"""

from __future__ import annotations

import csv
import json
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

from axia80_feasibility.measurement import (  # noqa: E402
    WRENCH_COMPONENTS,
    default_masses_g,
    evaluate_acceptance,
    read_summary_csv,
    regression_metrics,
    summarize_wrenches,
    write_sample_csv,
    write_summary_csv,
)
from axia80_feasibility.model import (  # noqa: E402
    ARM_POSE,
    SENSOR_HEIGHT_M,
    SENSOR_LINK_NAME,
    SENSOR_MASS_KG,
    SENSOR_RADIUS_M,
    TOOL_GAP_M,
    TOOL_LINK_NAME,
    TOOL_MASS_KG,
    TOOL_SIZE_M,
    _append_axia80_and_tool,
    build_model,
)
from axia80_feasibility.plotting import plot_summary  # noqa: E402


GRAVITY = 9.81


def _synthetic_measurement() -> tuple[list[int], np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Create deterministic raw samples with a known tare and payload response."""
    masses_g = [10, 50, 100, 250, 500]
    env_count = len(masses_g)
    mass_kg = np.asarray(masses_g, dtype=float) * 1.0e-3

    tare_center = np.tile(
        np.asarray([0.021, -0.034, TOOL_MASS_KG * GRAVITY, 0.004, -0.006, 0.002]),
        (env_count, 1),
    )
    tare_center += np.arange(env_count, dtype=float)[:, None] * np.asarray(
        [1.0e-4, -2.0e-4, 3.0e-4, 1.0e-5, -1.0e-5, 2.0e-5]
    )
    tare_noise = np.asarray([2.0e-3, 1.0e-3, 3.0e-3, 2.0e-4, 1.0e-4, 1.5e-4])
    tare_samples = tare_center[None, :, :] + np.asarray([-1.0, 0.0, 1.0])[:, None, None] * tare_noise

    response = np.zeros((env_count, 6), dtype=float)
    response[:, 0] = 0.008 * mass_kg
    response[:, 1] = -0.006 * mass_kg
    response[:, 2] = GRAVITY * mass_kg
    response[:, 3] = 8.0e-4 * mass_kg
    response[:, 4] = -6.0e-4 * mass_kg
    response[:, 5] = 2.0e-4 * mass_kg
    sample_noise = np.asarray([1.0e-3, 1.5e-3, 2.0e-3, 1.0e-4, 2.0e-4, 1.0e-4])
    noise_factors = np.asarray([-1.5, -0.5, 0.5, 1.5])
    samples = (
        tare_center[None, :, :]
        + response[None, :, :]
        + noise_factors[:, None, None] * sample_noise
    )

    rows = summarize_wrenches(
        masses_g,
        samples,
        tare_samples,
        gravity_m_s2=GRAVITY,
        settled=[True] * env_count,
        settle_time_s=1.75,
        sample_max_relative_speed_m_s=[8.0e-4] * env_count,
        sample_max_relative_angular_speed_rad_s=[4.0e-3] * env_count,
        tool_normal_z=[1.0] * env_count,
    )
    return masses_g, samples, tare_samples, rows


def _numbers(element: ET.Element, attribute: str) -> np.ndarray:
    return np.fromstring(element.get(attribute, ""), sep=" ")


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _joint_transform(joint: ET.Element, positions: dict[str, float]) -> np.ndarray:
    transform = np.eye(4)
    origin = joint.find("origin")
    if origin is not None:
        transform[:3, 3] = _numbers(origin, "xyz")
        transform[:3, :3] = _rpy_matrix(_numbers(origin, "rpy"))
    if joint.get("type") in ("continuous", "revolute"):
        motion = np.eye(4)
        motion[:3, :3] = _axis_angle_matrix(
            _numbers(joint.find("axis"), "xyz"), positions[joint.get("name")]
        )
        transform = transform @ motion
    return transform


def _link_transform(robot: ET.Element, link_name: str, positions: dict[str, float]) -> np.ndarray:
    joints_by_child = {
        joint.find("child").get("link"): joint for joint in robot.findall("joint")
    }
    chain: list[ET.Element] = []
    seen: set[str] = set()
    while link_name in joints_by_child:
        if link_name in seen:
            raise AssertionError(f"joint cycle at {link_name}")
        seen.add(link_name)
        joint = joints_by_child[link_name]
        chain.append(joint)
        link_name = joint.find("parent").get("link")
    transform = np.eye(4)
    for joint in reversed(chain):
        transform = transform @ _joint_transform(joint, positions)
    return transform


class MassSequenceTests(unittest.TestCase):
    def test_default_sequence_is_inclusive_10_to_500_grams(self) -> None:
        masses = default_masses_g()
        self.assertEqual(masses, list(range(10, 501, 10)))
        self.assertEqual(len(masses), 50)
        self.assertTrue(all(right > left for left, right in zip(masses, masses[1:])))

    def test_custom_sequence_and_invalid_ranges(self) -> None:
        self.assertEqual(default_masses_g(25, 100, 25), [25, 50, 75, 100])
        for arguments in ((0, 100, 10), (100, 50, 10), (10, 100, 0), (10, 95, 10)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                default_masses_g(*arguments)


class MeasurementReductionTests(unittest.TestCase):
    def test_synthetic_tare_reduction_regression_and_acceptance(self) -> None:
        masses_g, samples, tare_samples, rows = _synthetic_measurement()

        self.assertEqual([row["mass_g"] for row in rows], masses_g)
        self.assertEqual([row["env_id"] for row in rows], list(range(len(masses_g))))
        for env_id, row in enumerate(rows):
            expected_fz = masses_g[env_id] * 1.0e-3 * GRAVITY
            self.assertAlmostEqual(float(row["Fz_N"]), expected_fz, places=12)
            self.assertAlmostEqual(float(row["Fz_error_N"]), 0.0, places=12)
            self.assertAlmostEqual(float(row["Fz_error_percent"]), 0.0, places=10)
            self.assertEqual(row["sample_count"], samples.shape[0])
            self.assertEqual(row["tare_sample_count"], tare_samples.shape[0])
            self.assertEqual(row["sample_max_relative_speed_m_s"], 8.0e-4)
            self.assertEqual(row["sample_max_relative_angular_speed_rad_s"], 4.0e-3)
            self.assertNotIn("final_speed_m_s", row)
            self.assertNotIn("final_angular_speed_rad_s", row)
            self.assertAlmostEqual(
                float(row["raw_Fz_N"]) - float(row["tare_Fz_N"]),
                expected_fz,
                places=12,
            )
            self.assertGreater(float(row["Fz_std_N"]), 0.0)
            self.assertGreater(float(row["tare_Fz_std_N"]), 0.0)

        expected_tare_fz = TOOL_MASS_KG * GRAVITY
        metrics = regression_metrics(rows, expected_tare_fz_n=expected_tare_fz)
        self.assertAlmostEqual(float(metrics["slope_N_per_kg"]), GRAVITY, places=12)
        self.assertAlmostEqual(float(metrics["intercept_N"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["r_squared"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["fz_rmse_N"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["max_abs_fz_error_N"]), 0.0, places=12)
        self.assertTrue(metrics["fz_monotonic"])
        self.assertAlmostEqual(float(metrics["mean_tare_fz_N"]), expected_tare_fz + 6.0e-4)
        self.assertAlmostEqual(float(metrics["max_abs_tare_fz_error_N"]), 1.2e-3)
        self.assertAlmostEqual(float(metrics["tare_fz_span_N"]), 1.2e-3)
        self.assertTrue(metrics["all_settled"])
        self.assertEqual(metrics["min_tool_normal_z"], 1.0)
        passed, failures = evaluate_acceptance(metrics, gravity_m_s2=GRAVITY)
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_acceptance_reports_every_failed_contract(self) -> None:
        _, _, _, rows = _synthetic_measurement()
        bad_metrics = dict(
            regression_metrics(rows, expected_tare_fz_n=TOOL_MASS_KG * GRAVITY)
        )
        bad_metrics.update(
            {
                "slope_N_per_kg": 8.0,
                "r_squared": 0.9,
                "intercept_N": 0.03,
                "fz_rmse_N": 0.02,
                "max_abs_fz_error_N": 0.03,
                "fz_monotonic": False,
                "max_abs_tare_fz_error_N": 0.03,
                "max_cross_axis_force_N": 0.2,
                "max_abs_moment_Nm": 0.03,
                "all_settled": False,
                "min_tool_normal_z": 0.995,
            }
        )
        passed, failures = evaluate_acceptance(bad_metrics, gravity_m_s2=GRAVITY)
        self.assertFalse(passed)
        self.assertEqual(len(failures), 11)
        for fragment in (
            "slope",
            "R^2",
            "intercept",
            "RMSE",
            "Fz-m*g",
            "strictly increasing",
            "tare Fz error",
            "cross-axis",
            "moment",
            "stability",
            "alignment",
        ):
            with self.subTest(fragment=fragment):
                self.assertTrue(any(fragment in failure for failure in failures))

    def test_acceptance_rejects_constant_fz_bias_and_non_finite_metrics(self) -> None:
        _, _, _, rows = _synthetic_measurement()
        biased_rows = [dict(row) for row in rows]
        for row in biased_rows:
            row["Fz_N"] = float(row["Fz_N"]) + 0.05
        biased_metrics = regression_metrics(
            biased_rows, expected_tare_fz_n=TOOL_MASS_KG * GRAVITY
        )
        self.assertAlmostEqual(float(biased_metrics["slope_N_per_kg"]), GRAVITY, places=12)
        self.assertAlmostEqual(float(biased_metrics["r_squared"]), 1.0, places=12)
        passed, failures = evaluate_acceptance(biased_metrics, gravity_m_s2=GRAVITY)
        self.assertFalse(passed)
        self.assertTrue(any("intercept" in failure for failure in failures))
        self.assertTrue(any("RMSE" in failure for failure in failures))
        self.assertTrue(any("Fz-m*g" in failure for failure in failures))

        non_finite_metrics = dict(biased_metrics)
        non_finite_metrics["slope_N_per_kg"] = math.nan
        passed, failures = evaluate_acceptance(non_finite_metrics, gravity_m_s2=GRAVITY)
        self.assertFalse(passed)
        self.assertIn("one or more acceptance metrics are non-finite", failures)

    def test_reduction_rejects_bad_shapes_short_windows_and_non_finite_data(self) -> None:
        masses_g = [10, 20]
        keyword_arguments = {
            "gravity_m_s2": GRAVITY,
            "settled": [True, True],
            "settle_time_s": 1.0,
            "sample_max_relative_speed_m_s": [0.0, 0.0],
            "sample_max_relative_angular_speed_rad_s": [0.0, 0.0],
            "tool_normal_z": [1.0, 1.0],
        }
        valid = np.zeros((2, 2, 6))
        with self.assertRaises(ValueError):
            summarize_wrenches(masses_g, np.zeros((2, 1, 6)), valid, **keyword_arguments)
        with self.assertRaises(ValueError):
            summarize_wrenches(masses_g, valid[:1], valid, **keyword_arguments)
        invalid = valid.copy()
        invalid[0, 0, 2] = np.nan
        with self.assertRaises(ValueError):
            summarize_wrenches(masses_g, invalid, valid, **keyword_arguments)
        with self.assertRaises(ValueError):
            regression_metrics([{"mass_kg": 0.1, "Fz_N": 1.0}])


class CsvAndPlotTests(unittest.TestCase):
    def test_summary_csv_round_trip_preserves_values_and_types(self) -> None:
        _, _, _, rows = _synthetic_measurement()
        with tempfile.TemporaryDirectory(prefix="axia80-csv-") as directory:
            path = Path(directory) / "nested" / "summary.csv"
            write_summary_csv(path, rows)
            restored = read_summary_csv(path)

        self.assertEqual(restored, rows)
        self.assertIsInstance(restored[0]["env_id"], int)
        self.assertIsInstance(restored[0]["mass_g"], int)
        self.assertIsInstance(restored[0]["settled"], bool)
        self.assertIsInstance(restored[0]["Fz_N"], float)

    def test_sample_csv_contains_raw_and_tare_corrected_wrenches(self) -> None:
        masses_g, samples, tare_samples, _ = _synthetic_measurement()
        tare_mean = tare_samples.mean(axis=0)
        with tempfile.TemporaryDirectory(prefix="axia80-samples-") as directory:
            path = Path(directory) / "samples.csv"
            write_sample_csv(path, masses_g, samples, tare_mean, dt_s=0.004)
            with path.open(newline="") as stream:
                records = list(csv.DictReader(stream))

        self.assertEqual(len(records), samples.shape[0] * len(masses_g))
        self.assertEqual(
            list(records[0]),
            [
                "sample_index",
                "time_s",
                "env_id",
                "mass_g",
                "Fx_N",
                "Fy_N",
                "Fz_N",
                "Mx_Nm",
                "My_Nm",
                "Mz_Nm",
                "raw_Fx_N",
                "raw_Fy_N",
                "raw_Fz_N",
                "raw_Mx_Nm",
                "raw_My_Nm",
                "raw_Mz_Nm",
            ],
        )
        first = records[0]
        self.assertEqual((first["sample_index"], first["env_id"], first["mass_g"]), ("0", "0", "10"))
        for index, component in enumerate(WRENCH_COMPONENTS):
            unit = "N" if index < 3 else "Nm"
            self.assertAlmostEqual(float(first[f"raw_{component}_{unit}"]), samples[0, 0, index])
            self.assertAlmostEqual(
                float(first[f"{component}_{unit}"]),
                samples[0, 0, index] - tare_mean[0, index],
            )

    def test_csv_guards_empty_unsorted_and_non_finite_summaries(self) -> None:
        masses_g, samples, tare_samples, rows = _synthetic_measurement()
        with tempfile.TemporaryDirectory(prefix="axia80-invalid-csv-") as directory:
            directory_path = Path(directory)
            with self.assertRaises(ValueError):
                write_summary_csv(directory_path / "empty.csv", [])

            unsorted_path = directory_path / "unsorted.csv"
            write_summary_csv(unsorted_path, list(reversed(rows)))
            with self.assertRaises(ValueError):
                read_summary_csv(unsorted_path)

            invalid_rows = [dict(row) for row in rows]
            invalid_rows[0]["Fz_N"] = math.inf
            invalid_path = directory_path / "invalid.csv"
            write_summary_csv(invalid_path, invalid_rows)
            with self.assertRaises(ValueError):
                read_summary_csv(invalid_path)

            duplicate_path = directory_path / "duplicate.csv"
            write_summary_csv(duplicate_path, [rows[0], rows[0]])
            with self.assertRaises(ValueError):
                read_summary_csv(duplicate_path)

            with self.assertRaises(ValueError):
                write_sample_csv(
                    directory_path / "invalid-dt.csv",
                    masses_g,
                    samples,
                    tare_samples.mean(axis=0),
                    dt_s=0.0,
                )

    def test_plot_summary_creates_a_nonempty_png(self) -> None:
        _, _, _, rows = _synthetic_measurement()
        with tempfile.TemporaryDirectory(prefix="axia80-plot-") as directory:
            directory_path = Path(directory)
            csv_path = directory_path / "summary.csv"
            output_path = directory_path / "plots" / "response.png"
            write_summary_csv(csv_path, rows)
            returned_path = plot_summary(csv_path, output_path, dpi=60)
            png = output_path.read_bytes()

        self.assertEqual(returned_path, output_path)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(png), 10_000)
        self.assertGreater(int.from_bytes(png[16:20], "big"), 500)
        self.assertGreater(int.from_bytes(png[20:24], "big"), 300)


class ModelContractTests(unittest.TestCase):
    def test_axia80_and_tool_geometry_share_an_unambiguous_measurement_frame(self) -> None:
        robot = ET.Element("robot", {"name": "geometry_contract"})
        ET.SubElement(robot, "link", {"name": "tool0"})
        _append_axia80_and_tool(robot)

        sensor = robot.find(f"link[@name='{SENSOR_LINK_NAME}']")
        tool = robot.find(f"link[@name='{TOOL_LINK_NAME}']")
        self.assertIsNotNone(sensor)
        self.assertIsNotNone(tool)
        for role in ("visual", "collision"):
            cylinder = sensor.find(f"{role}/geometry/cylinder")
            self.assertAlmostEqual(float(cylinder.get("radius")), 0.041)
            self.assertAlmostEqual(float(cylinder.get("length")), 0.0254)
            np.testing.assert_allclose(_numbers(sensor.find(f"{role}/origin"), "xyz"), 0.0)

            box = tool.find(f"{role}/geometry/box")
            size = _numbers(box, "size")
            np.testing.assert_allclose(size, TOOL_SIZE_M)
            self.assertGreater(size[0], size[2])
            self.assertGreater(size[1], size[2])

        sensor_mass = float(sensor.find("inertial/mass").get("value"))
        self.assertAlmostEqual(sensor_mass, SENSOR_MASS_KG)
        sensor_inertia = sensor.find("inertial/inertia")
        radial = SENSOR_MASS_KG * (3.0 * SENSOR_RADIUS_M**2 + SENSOR_HEIGHT_M**2) / 12.0
        axial = 0.5 * SENSOR_MASS_KG * SENSOR_RADIUS_M**2
        self.assertAlmostEqual(float(sensor_inertia.get("ixx")), radial)
        self.assertAlmostEqual(float(sensor_inertia.get("iyy")), radial)
        self.assertAlmostEqual(float(sensor_inertia.get("izz")), axial)

        mount = robot.find("joint[@name='ur5e_to_axia80_joint']")
        self.assertEqual(mount.get("type"), "fixed")
        self.assertEqual(mount.find("parent").get("link"), "tool0")
        self.assertEqual(mount.find("child").get("link"), SENSOR_LINK_NAME)
        np.testing.assert_allclose(
            _numbers(mount.find("origin"), "xyz"), (0.0, 0.0, 0.5 * SENSOR_HEIGHT_M)
        )

        measurement = robot.find("joint[@name='axia80_measurement_joint']")
        self.assertEqual(measurement.get("type"), "fixed")
        self.assertEqual(measurement.find("parent").get("link"), SENSOR_LINK_NAME)
        self.assertEqual(measurement.find("child").get("link"), TOOL_LINK_NAME)
        np.testing.assert_allclose(_numbers(measurement.find("origin"), "xyz"), 0.0)
        np.testing.assert_allclose(_numbers(measurement.find("origin"), "rpy"), 0.0)

        expected_tool_center = 0.5 * SENSOR_HEIGHT_M + TOOL_GAP_M + 0.5 * TOOL_SIZE_M[2]
        for origin in (
            tool.find("inertial/origin"),
            tool.find("visual/origin"),
            tool.find("collision/origin"),
        ):
            np.testing.assert_allclose(_numbers(origin, "xyz"), (0.0, 0.0, expected_tool_center))

    def test_model_build_emits_valid_metadata_and_an_upward_tool_normal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="axia80-model-") as directory:
            output = Path(directory) / "generated"
            urdf_path, metadata = build_model(WORKSPACE_ROOT, output)
            first_content = urdf_path.read_bytes()
            second_path, second_metadata = build_model(WORKSPACE_ROOT, output)
            robot = ET.parse(urdf_path).getroot()
            json_metadata = json.loads((output / "model.json").read_text())

            self.assertEqual(second_path, urdf_path)
            self.assertEqual(second_metadata, metadata)
            self.assertEqual(urdf_path.read_bytes(), first_content)
            for mesh in robot.iter("mesh"):
                mesh_path = Path(mesh.get("filename"))
                self.assertTrue(mesh_path.is_absolute())
                self.assertTrue(mesh_path.is_file(), mesh_path)

        self.assertEqual(metadata["sensor_radius_m"], 0.041)
        self.assertEqual(metadata["sensor_height_m"], 0.0254)
        self.assertEqual(metadata["wrench_order"], ["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
        self.assertEqual(json_metadata["sensor_link_name"], SENSOR_LINK_NAME)
        self.assertEqual(json_metadata["tool_link_name"], TOOL_LINK_NAME)
        self.assertAlmostEqual(
            float(metadata["tool_top_z_in_sensor_m"]),
            0.5 * SENSOR_HEIGHT_M + TOOL_GAP_M + TOOL_SIZE_M[2],
        )

        sensor_transform = _link_transform(robot, SENSOR_LINK_NAME, ARM_POSE)
        tool_transform = _link_transform(robot, TOOL_LINK_NAME, ARM_POSE)
        np.testing.assert_allclose(tool_transform, sensor_transform, atol=1.0e-9)
        world_tool_normal = tool_transform[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
        np.testing.assert_allclose(world_tool_normal, (0.0, 0.0, 1.0), atol=1.0e-8)

    def test_geometry_builder_requires_the_ur_tool_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool0"):
            _append_axia80_and_tool(ET.Element("robot"))


if __name__ == "__main__":
    unittest.main()
