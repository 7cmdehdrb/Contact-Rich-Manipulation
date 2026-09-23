import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inspire_tactile.projected_tactile import assign_regions, ProjectedTactileReader
from inspire_tactile.regions import load_regions
from inspire_tactile.model import repo_root


BOX = np.array([[1, 0, 0, -1], [-1, 0, 0, -1], [0, 1, 0, -1],
                [0, -1, 0, -1], [0, 0, 1, -1], [0, 0, -1, -1]], dtype=np.float32)


class RegionTest(unittest.TestCase):
    def test_boundary_tolerance_and_outside_not_nearest(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1.001, 0, 0], [1.003, 0, 0], [10, 0, 0]])
        args = (points, np.zeros((1, 3)), np.array([[1, 0, 0, 0]]), [BOX])
        self.assertEqual(assign_regions(*args, 0).tolist(), [0, 0, -1, -1, -1])
        self.assertEqual(assign_regions(*args, .002).tolist(), [0, 0, 0, -1, -1])

    def test_pose_and_overlap_single_assignment(self):
        thin = BOX.copy()
        thin[2:, 3] = -.1
        q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        points = np.array([[3, 4.9, 5], [3.9, 4, 5]])
        selected = assign_regions(points, np.array([[3, 4, 5]] * 2), np.array([q] * 2), [thin, thin], 0)
        self.assertEqual(selected.tolist(), [0, -1])  # overlap tie is deterministic, not duplicated

    def fixture(self):
        sensor = SimpleNamespace(body_names=["a", "b"], cfg=SimpleNamespace(debug_vis=False),
                                 data=SimpleNamespace(net_forces_w=torch.zeros(2, 2, 3),
                                                      pos_w=torch.tensor([[[0., 0, 0]] * 2, [[10., 0, 0]] * 2]),
                                                      quat_w=torch.tensor([[[1., 0, 0, 0]] * 2] * 2)))
        sensor.data.net_forces_w[0, 0, 0] = .01
        scalars = torch.full((8, 1), float("nan"))
        points = torch.full((8, 3), float("nan"))
        normals = torch.full((8, 3), float("nan"))
        scalars[:3, 0] = torch.tensor([2., 3., 4.])
        points[:3] = torch.tensor([[0., 0, 0], [3., 0, 0], [10., 0, 0]])
        normals[:3] = torch.tensor([[1., 0, 0]] * 3)
        buffers = (scalars, points, normals, torch.zeros(8, 1), torch.tensor([[2], [1]]), torch.tensor([[0], [2]]))
        matrix = torch.tensor([[[[5., 0, 0]]], [[[4., 0, 0]]]])
        observer = SimpleNamespace(body_names=["parent"], data=SimpleNamespace(force_matrix_w=matrix,
                                                                             net_forces_w=matrix.sum(2)),
                                   contact_physx_view=SimpleNamespace(get_contact_data=lambda dt: buffers))
        with patch("inspire_tactile.projected_tactile.load_regions", return_value={"parent": [(0, BOX), (1, BOX)]}):
            reader = ProjectedTactileReader({"tactile": sensor, "contact_region_parent": observer}, Path("unused"), ["a", "b"])
        return reader, observer, buffers

    def test_real_points_only_batch_raw_preserved_and_force_conservation(self):
        reader, _, _ = self.fixture()
        force, norm, bits = reader.read()
        np.testing.assert_allclose(force[:, 0, 0], [2.01, 4])
        self.assertEqual(bits.tolist(), [[1, 0], [1, 0]])
        np.testing.assert_allclose(reader.unmapped_forces_w, [[3, 0, 0], [0, 0, 0]])
        self.assertEqual(reader.unmapped_count.tolist(), [1, 0])
        np.testing.assert_allclose(reader.mapped_forces_w.sum(1) + reader.unmapped_forces_w,
                                   [[5, 0, 0], [4, 0, 0]])
        # Unused NaN buffer slots must not be treated as contacts.
        self.assertTrue(torch.isfinite(norm).all())

    def test_no_contacts_zero_and_invalid_buffers_fail_loudly(self):
        reader, observer, buffers = self.fixture()
        buffers[4].zero_()
        observer.data.force_matrix_w.zero_()
        observer.data.net_forces_w.zero_()
        reader.sensor.data.net_forces_w.zero_()
        self.assertEqual(reader.read()[2].sum().item(), 0)
        for invalid in ("overflow", "missing", "nonfinite", "mismatch"):
            reader, observer, buffers = self.fixture()
            if invalid == "overflow":
                buffers[4][0, 0] = 8
            elif invalid == "missing":
                buffers[4][0, 0] = 0
            elif invalid == "nonfinite":
                buffers[1][0, 0] = float("nan")
            else:
                observer.data.force_matrix_w[0, 0, 0, 0] = 6
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                reader.read()

    def test_unfiltered_forces_not_fabricated_as_pad_contacts(self):
        reader, observer, _ = self.fixture()
        observer.data.force_matrix_w = None
        reader.sensor.data.net_forces_w.zero_()
        self.assertEqual(reader.read()[2].sum().item(), 0)
        np.testing.assert_allclose(reader.unobserved_forces_w, [[5, 0, 0], [4, 0, 0]])
        np.testing.assert_allclose(reader.unmapped_forces_w, reader.unobserved_forces_w)

    def test_invalid_sensor_pose_is_not_silently_classified_off(self):
        reader, _, _ = self.fixture()
        reader.sensor.data.quat_w.zero_()
        with self.assertRaises(RuntimeError):
            reader.read()

    def test_sensor_meshes_loaded_without_modification(self):
        # Use original local sensor names: same parser supports generated URDF.
        import hashlib
        import xml.etree.ElementTree as ET
        path = repo_root() / "src/inspire_tactile/generated/ur5e_inspire.urdf"
        if not path.exists():
            self.skipTest("Build the generated model first")
        digest = hashlib.sha256(path.read_bytes()).digest()
        names = [l.get("name") for l in ET.parse(path).getroot().findall("link") if "force_sensor" in l.get("name")]
        regions = load_regions(path, names)
        self.assertEqual(sorted(i for group in regions.values() for i, _ in group), list(range(17)))
        self.assertTrue(all(np.isfinite(eq).all() for group in regions.values() for _, eq in group))
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)


if __name__ == "__main__":
    unittest.main()
