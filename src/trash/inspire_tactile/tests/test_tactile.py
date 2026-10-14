"""Run with python -m unittest discover -s src/inspire_tactile/tests -v."""

import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from inspire_tactile.controller import CYCLE_SECONDS, phase_at
from inspire_tactile.model import build_model, repo_root, origin_matrix, CLOSED
from inspire_tactile.placement import object_and_support_positions
from inspire_tactile.tactile import TactileReader, binary_contact


class ContactContractTest(unittest.TestCase):
    def test_threshold_inclusive_and_vector_norm(self):
        force = torch.tensor([[[0.0, 0.0, 0.0], [0.009, 0.0, 0.0],
                               [0.01, 0.0, 0.0], [-0.006, 0.008, 0.0], [0.0, 0.0, -0.02]]])
        magnitude, bits = binary_contact(force)
        self.assertEqual(bits.tolist(), [[0, 0, 1, 1, 1]])
        self.assertEqual(bits.dtype, torch.uint8)
        self.assertEqual(magnitude.shape, (1, 5))

    def test_invalid_data_does_not_look_like_no_contact(self):
        with self.assertRaises(RuntimeError):
            binary_contact(torch.tensor([[float("nan"), 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            binary_contact(torch.zeros(1, 17, 2))

    def test_name_ordering_and_batch_shape(self):
        sensor = SimpleNamespace(body_names=["b", "a"], data=SimpleNamespace(
            net_forces_w=torch.tensor([[[0.03, 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0.02, 0., 0.]]])))
        reader = TactileReader(sensor, ["a", "b"])
        forces, magnitude, bits = reader.read()
        self.assertEqual(bits.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(forces.shape, (2, 2, 3))
        with self.assertRaises(RuntimeError):
            TactileReader(sensor, ["a", "missing"])

    def test_open_close_release_schedule(self):
        self.assertEqual(phase_at(0), ("open", 0.0))
        self.assertEqual(phase_at(5), ("hold", 1.0))
        self.assertEqual(phase_at(10), ("released", 0.0))
        self.assertEqual(phase_at(CYCLE_SECONDS), ("open", 0.0))


class AssetContractTest(unittest.TestCase):
    def test_world_lowering_keeps_support_under_cylinder(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        old_object, old_support = object_and_support_positions(pose, lowering=0)
        new_object, new_support = object_and_support_positions(pose)
        np.testing.assert_allclose(new_object - old_object, [0, 0, -0.02])
        np.testing.assert_allclose(new_support - old_support, [0, 0, -0.02])
        self.assertAlmostEqual(new_object[2] - 0.09 / 2, new_support[2] + 0.04 / 2)
        for invalid in (-0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                object_and_support_positions(pose, lowering=invalid)

    def test_original_preserved_and_sensor_bodies_retained(self):
        root = repo_root()
        original = root / "src/inspire_robot/urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf"
        digest = hashlib.sha256(original.read_bytes()).digest()
        with tempfile.TemporaryDirectory() as folder:
            path, model = build_model(root, Path(folder))
            robot = ET.parse(path).getroot()
            # tool0 and wrist_3_link coincide in the UR description. Hand Z must
            # stay coaxial after massless-frame collapse, not bend by 90 degrees.
            mount = robot.find("joint[@name='inspire_mount']")
            self.assertEqual(mount.find("parent").get("link"), "wrist_3_link")
            transform = origin_matrix(mount.find("origin"))
            np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-10)
            np.testing.assert_allclose(transform[:3, 3], [0, 0, 0], atol=1e-10)
            np.testing.assert_allclose(np.array(model["hand_pose"])[:3, :3],
                                       [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-8)
            np.testing.assert_allclose(np.array(model["hand_pose"])[:3, 3],
                                       [0.4919, 0.1333, 0.4629], atol=1e-8)
            self.assertEqual(robot.findall(".//mimic"), [])  # no duplicate PhysX mimic + drive constraints
            self.assertEqual(len(model["sensor_names"]), 17)
            links = {link.get("name"): link for link in robot.findall("link")}
            joints = {j.find("child").get("link"): j for j in robot.findall("joint")}
            for name in model["sensor_names"]:
                self.assertIsNotNone(links[name].find("collision"))
                self.assertIsNotNone(links[name].find("inertial"))
                self.assertEqual(joints[name].get("type"), "fixed")
            for mesh in robot.iter("mesh"):
                self.assertTrue(Path(mesh.get("filename")).is_file())
            self.assertEqual(len(model["open_targets"]), 12)
            self.assertAlmostEqual(model["closed_targets"]["inspire_left_index_2_joint"], 1.4381 * 1.0843)
            original_hand = ET.parse(original).getroot()
            for name, target in CLOSED.items():
                upper = float(original_hand.find(f"joint[@name='{name}']/limit").get("upper"))
                self.assertAlmostEqual(target, upper)
            mtime = path.stat().st_mtime_ns
            build_model(root, Path(folder))
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.assertEqual(hashlib.sha256(original.read_bytes()).digest(), digest)


if __name__ == "__main__":
    unittest.main()
