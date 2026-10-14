"""Pure-torch tests for the standalone sensor contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_manipulation_rl.sensors import (  # noqa: E402
    BilateralTactileReader,
    DORSAL_CHANNEL_NAMES,
    DORSAL_PARENT_BODY_NAMES,
    DORSAL_PARENT_BY_CHANNEL,
    DORSAL_PARENT_MAP,
    FixedJointWrenchReader,
    PALM_CHANNEL_NAMES,
    contact_magnitudes_and_bits,
    hand_object_resultant_normal,
    robot_board_contact_forces,
    transform_wrench_f_to_c,
    weighted_hand_object_resultant_normal,
)


def sensor(body_names, forces=None, force_matrix=None):
    data = SimpleNamespace()
    if forces is not None:
        data.net_forces_w = forces
    if force_matrix is not None or hasattr(data, "force_matrix_w"):
        data.force_matrix_w = force_matrix
    return SimpleNamespace(body_names=list(body_names), data=data)


class TactileContractTest(unittest.TestCase):
    def test_explicit_channel_order_and_dorsal_mapping(self):
        self.assertEqual(len(PALM_CHANNEL_NAMES), 17)
        self.assertEqual(len(DORSAL_CHANNEL_NAMES), 17)
        self.assertEqual(len(DORSAL_PARENT_BY_CHANNEL), 17)
        self.assertEqual(PALM_CHANNEL_NAMES[0], "inspire_palm_force_sensor")
        self.assertEqual(PALM_CHANNEL_NAMES[1:5], tuple(
            f"inspire_thumb_force_sensor_{index}" for index in range(1, 5)
        ))
        self.assertEqual(DORSAL_PARENT_BY_CHANNEL[0], "inspire_base_link")
        # Two distal regions intentionally share one parent-body approximation.
        self.assertEqual(DORSAL_PARENT_BY_CHANNEL[3], DORSAL_PARENT_BY_CHANNEL[4])
        self.assertEqual(DORSAL_PARENT_BY_CHANNEL[6], DORSAL_PARENT_BY_CHANNEL[7])
        self.assertEqual(len(DORSAL_PARENT_BODY_NAMES), 12)
        self.assertEqual(
            DORSAL_PARENT_MAP["inspire_dorsal_palm_force_sensor"], "inspire_base_link"
        )

    def test_binary_threshold_is_inclusive_and_rejects_invalid_data(self):
        forces = torch.tensor([[[0.03, 0.04, 0.0], [0.049, 0.0, 0.0]]])
        magnitudes, bits = contact_magnitudes_and_bits(forces, 0.05)
        torch.testing.assert_close(magnitudes, torch.tensor([[0.05, 0.049]]))
        self.assertEqual(bits.dtype, torch.uint8)
        self.assertEqual(bits.tolist(), [[1, 0]])
        with self.assertRaises(ValueError):
            contact_magnitudes_and_bits(forces, 0.0)
        bad = forces.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            contact_magnitudes_and_bits(bad, 0.05)

    def test_bilateral_reader_reorders_palm_and_maps_dorsal_parents(self):
        palm_order = tuple(reversed(PALM_CHANNEL_NAMES))
        palm_forces = torch.zeros(2, 17, 3)
        for body_index, body_name in enumerate(palm_order):
            canonical_index = PALM_CHANNEL_NAMES.index(body_name)
            palm_forces[:, body_index, 0] = canonical_index / 100.0

        # Shuffle the 12 required parents and include one harmless extra robot body.
        parent_order = tuple(reversed(DORSAL_PARENT_BODY_NAMES)) + ("unrelated_arm_link",)
        dorsal_forces = torch.zeros(2, len(parent_order), 3)
        for body_index, body_name in enumerate(parent_order):
            if body_name in DORSAL_PARENT_BODY_NAMES:
                canonical_index = DORSAL_PARENT_BODY_NAMES.index(body_name)
                dorsal_forces[:, body_index, 1] = 0.2 + canonical_index / 100.0

        reader = BilateralTactileReader(
            sensor(palm_order, palm_forces),
            sensor(parent_order, dorsal_forces),
            threshold_n=0.10,
            dorsal_threshold_n=0.25,
        )
        sample = reader.read()
        self.assertEqual(sample.forces_w.shape, (2, 34, 3))
        self.assertEqual(sample.bits.shape, (2, 34))
        # Palm was reversed in the sensor but is restored to canonical IDs.
        torch.testing.assert_close(
            sample.palm_forces_w[0, :, 0], torch.arange(17, dtype=torch.float32) / 100.0
        )
        self.assertEqual(sample.palm_bits[0].tolist(), [0] * 10 + [1] * 7)
        # Shared distal parent means the two logical dorsal channels match.
        self.assertEqual(
            sample.dorsal_forces_w[0, 3].tolist(), sample.dorsal_forces_w[0, 4].tolist()
        )
        self.assertEqual(sample.dorsal_bits[0, 3].item(), sample.dorsal_bits[0, 4].item())

    def test_bilateral_reader_fails_on_wrong_physical_palm_set(self):
        names = list(PALM_CHANNEL_NAMES)
        names[-1] = "wrong"
        with self.assertRaises(RuntimeError):
            BilateralTactileReader(
                sensor(names, torch.zeros(1, 17, 3)),
                sensor(DORSAL_PARENT_BODY_NAMES, torch.zeros(1, 12, 3)),
            )


class WrenchContractTest(unittest.TestCase):
    def test_f_to_c_rotation_and_moment_arm(self):
        # +90 deg about C-z maps F+x to C+y and F+y moment to C-x negative.
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        wrench_f = torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
        transformed = transform_wrench_f_to_c(
            wrench_f, rotation, torch.tensor([1.0, 0.0, 0.0])
        )
        # Rotated moment is [-2,0,0]; r x F contributes [0,0,1].
        torch.testing.assert_close(
            transformed, torch.tensor([[0.0, 1.0, 0.0, -2.0, 0.0, 1.0]])
        )

    def test_child_body_lookup_keeps_self_weight_and_applies_only_frame_transform(self):
        incoming = torch.zeros(2, 3, 6)
        robot = SimpleNamespace(
            body_names=["base", "axia80_link", "inspire_base_link"],
            data=SimpleNamespace(body_incoming_joint_wrench_b=incoming),
        )
        # These non-zero forces/moments represent the Hand self-weight reaction
        # plus any external load.  They must never be tared or gravity-cancelled.
        self_weight_and_load = torch.tensor([
            [1.0, 2.0, 13.0, 0.1, 0.2, 0.3],
            [-1.0, 1.0, 12.0, -0.1, 0.3, 0.5],
        ])
        incoming[:, 2] = self_weight_and_load
        reader = FixedJointWrenchReader(robot, "inspire_base_link")
        self.assertEqual(reader.child_body_index, 2)
        identity = torch.eye(3).expand(2, -1, -1)
        sample = reader.read(identity, torch.zeros(2, 3))
        torch.testing.assert_close(sample.raw_f, self_weight_and_load)
        torch.testing.assert_close(sample.measured_f, self_weight_and_load)
        torch.testing.assert_close(sample.measured_c, self_weight_and_load)

    def test_reader_rejects_invalid_or_nonfinite_incoming_wrench(self):
        bad_shape = SimpleNamespace(
            body_names=["hand"],
            data=SimpleNamespace(body_incoming_joint_wrench_b=torch.zeros(2, 1, 5)),
        )
        with self.assertRaises(ValueError):
            FixedJointWrenchReader(bad_shape, "hand").read_raw_f()

        nonfinite = torch.zeros(2, 1, 6)
        nonfinite[0, 0, 0] = float("nan")
        bad_value = SimpleNamespace(
            body_names=["hand"],
            data=SimpleNamespace(body_incoming_joint_wrench_b=nonfinite),
        )
        with self.assertRaises(RuntimeError):
            FixedJointWrenchReader(bad_value, "hand").read_raw_f()


class ContactAggregationTest(unittest.TestCase):
    def test_board_force_sums_link_magnitudes_without_vector_cancellation(self):
        matrix = torch.tensor([[[
            [3.0, 4.0, 0.0],
            [-3.0, -4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]]])
        sample = robot_board_contact_forces(sensor(["board"], force_matrix=matrix))
        torch.testing.assert_close(sample.per_link_magnitudes_n, torch.tensor([[5.0, 5.0, 2.0]]))
        torch.testing.assert_close(sample.total_magnitude_n, torch.tensor([12.0]))
        # The first two vectors cancel, demonstrating why norm(sum(vectors)) is wrong.
        self.assertAlmostEqual(torch.linalg.vector_norm(matrix.sum(dim=(1, 2)), dim=-1).item(), 2.0)

    def test_board_filter_subset_and_missing_force_matrix(self):
        matrix = torch.ones(1, 2, 3, 3)
        sample = robot_board_contact_forces(
            sensor(["board_a", "board_b"], force_matrix=matrix),
            board_body_index=1,
            robot_filter_indices=[0, 2],
        )
        self.assertEqual(sample.pair_forces_w.shape, (1, 1, 2, 3))
        torch.testing.assert_close(
            sample.total_magnitude_n, torch.tensor([2.0 * (3.0**0.5)])
        )
        missing = SimpleNamespace(data=SimpleNamespace(force_matrix_w=None))
        with self.assertRaises(RuntimeError):
            robot_board_contact_forces(missing)

    def test_resultant_normal_uses_force_on_object_sign_and_handles_cancellation(self):
        # Inputs are reactions on the hand, so the force exerted on the object is +x.
        on_hand = torch.tensor([
            [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ])
        result = hand_object_resultant_normal(on_hand, min_resultant_force_n=0.01)
        torch.testing.assert_close(result.force_on_object_w[0], torch.tensor([3.0, 0.0, 0.0]))
        torch.testing.assert_close(result.normal_w[0], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(result.valid.tolist(), [True, False])
        torch.testing.assert_close(result.normal_w[1], torch.zeros(3))

    def test_weighted_normals_helper(self):
        normals_on_hand = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
        magnitudes = torch.tensor([[3.0, 4.0]])
        result = weighted_hand_object_resultant_normal(normals_on_hand, magnitudes)
        torch.testing.assert_close(result.magnitude_n, torch.tensor([5.0]))
        torch.testing.assert_close(result.normal_w, torch.tensor([[[0.6, 0.8, 0.0]]]).squeeze(1))


if __name__ == "__main__":
    unittest.main()
