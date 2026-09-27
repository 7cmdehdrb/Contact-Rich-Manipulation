"""Pure-torch checks for control-frame and Inspire action math.

The simulator-facing action classes intentionally require an initialized Isaac
runtime.  These tests execute the constants and helper functions from the same
source file while excluding its Isaac imports and class definitions, allowing
the frame contract to be checked without launching Kit.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType
import unittest

import torch


def _load_action_helpers() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "hand_manipulation_rl" / "mdp" / "actions.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            break
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            selected.append(node)
    helper_module = ModuleType("hand_manipulation_action_math")
    helper_module.__file__ = str(path)
    executable = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(executable, str(path), "exec"), helper_module.__dict__)
    return helper_module


action_math = _load_action_helpers()


class ControlFrameMathTest(unittest.TestCase):
    def test_local_delta_is_rotated_and_right_multiplied(self):
        dtype = torch.float64
        half = torch.tensor(torch.pi / 4.0, dtype=dtype)
        current_quaternion = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]], dtype=dtype)
        target_position, target_quaternion = action_math.current_frame_delta_target(
            torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype),
            current_quaternion,
            torch.tensor([[0.1, 0.0, 0.0]], dtype=dtype),
            torch.tensor([[0.2, 0.0, 0.0]], dtype=dtype),
        )
        torch.testing.assert_close(target_position, torch.tensor([[1.0, 2.1, 3.0]], dtype=dtype))
        expected_delta = action_math.quaternion_from_rotation_vector(torch.tensor([[0.2, 0.0, 0.0]], dtype=dtype))
        expected_quaternion = action_math.quaternion_multiply(current_quaternion, expected_delta)
        torch.testing.assert_close(target_quaternion, expected_quaternion)

    def test_zero_rotation_vector_is_finite_identity(self):
        quaternion = action_math.quaternion_from_rotation_vector(torch.zeros(8, 3))
        self.assertTrue(torch.isfinite(quaternion).all())
        torch.testing.assert_close(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(8, 1))

    def test_offset_pose_jacobian_and_twist_use_one_point(self):
        quaternion_h_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        position_c_b, quaternion_c_b, r_hc_b = action_math.compose_fixed_offset_pose(
            torch.tensor([[0.5, -0.2, 0.7]]),
            quaternion_h_b,
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(position_c_b, torch.tensor([[1.5, -0.2, 0.7]]))
        torch.testing.assert_close(quaternion_c_b, quaternion_h_b)

        jacobian_h_b = torch.zeros(1, 6, 2)
        jacobian_h_b[0, 5, 0] = 1.0  # first dof: rotation about B +z
        jacobian_h_b[0, 0, 1] = 1.0  # second dof: translation along B +x
        jacobian_c_b = action_math.shift_jacobian_to_point(jacobian_h_b, r_hc_b)
        torch.testing.assert_close(jacobian_c_b[0, :3, 0], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(jacobian_c_b[0, :3, 1], torch.tensor([1.0, 0.0, 0.0]))

        twist_h_b = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        twist_c_b = action_math.shift_twist_to_point(twist_h_b, r_hc_b)
        torch.testing.assert_close(twist_c_b, torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 2.0]]))

    def test_physx_com_jacobian_is_shifted_from_com_not_link_origin(self):
        half = torch.tensor(torch.pi / 4.0)
        quaternion_h_b = torch.tensor(
            [[torch.cos(half), 0.0, 0.0, torch.sin(half)]]
        )
        # H->COM is +x in H, hence +y in B. H->C is +x in B.
        r_comc_b = action_math.com_to_target_offset(
            quaternion_h_b,
            torch.tensor([[0.2, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.0, 0.0]]),
        )
        torch.testing.assert_close(r_comc_b, torch.tensor([[0.5, -0.2, 0.0]]))

        jacobian_com_b = torch.zeros(1, 6, 1)
        jacobian_com_b[0, 5, 0] = 1.0
        jacobian_c_b = action_math.shift_jacobian_to_point(
            jacobian_com_b, r_comc_b
        )
        torch.testing.assert_close(
            jacobian_c_b[0, :3, 0], torch.tensor([0.2, 0.5, 0.0])
        )

    def test_offset_jacobian_matches_finite_difference(self):
        dtype = torch.float64
        theta = torch.tensor(0.37, dtype=dtype)
        epsilon = 1.0e-6

        def point(angle: torch.Tensor) -> torch.Tensor:
            quaternion = action_math.quaternion_from_rotation_vector(
                torch.stack((torch.zeros_like(angle), torch.zeros_like(angle), angle)).reshape(1, 3)
            )
            position, _, _ = action_math.compose_fixed_offset_pose(
                torch.zeros(1, 3, dtype=dtype),
                quaternion,
                torch.tensor([[0.13, -0.04, 0.02]], dtype=dtype),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
            )
            return position[0]

        quaternion = action_math.quaternion_from_rotation_vector(
            torch.tensor([[0.0, 0.0, theta]], dtype=dtype)
        )
        _, _, r_hc_b = action_math.compose_fixed_offset_pose(
            torch.zeros(1, 3, dtype=dtype),
            quaternion,
            torch.tensor([[0.13, -0.04, 0.02]], dtype=dtype),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        )
        jacobian_h_b = torch.zeros(1, 6, 1, dtype=dtype)
        jacobian_h_b[0, 5, 0] = 1.0
        analytic = action_math.shift_jacobian_to_point(jacobian_h_b, r_hc_b)[0, :3, 0]
        numeric = (point(theta + epsilon) - point(theta - epsilon)) / (2.0 * epsilon)
        torch.testing.assert_close(analytic, numeric, rtol=1.0e-7, atol=1.0e-9)


class InspireSynergyMathTest(unittest.TestCase):
    def test_endpoints_reproduce_exact_urdf_maps(self):
        synergy = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        targets = action_math.inspire_synergy_to_joint_positions(synergy)
        closed = torch.tensor(action_math.INSPIRE_CLOSED_JOINT_POSITIONS)
        opened = torch.tensor(action_math.INSPIRE_OPEN_JOINT_POSITIONS)
        torch.testing.assert_close(targets[0], closed)
        torch.testing.assert_close(targets[1], opened)
        self.assertAlmostEqual(float(closed[2]), 0.5864 * 0.8024, places=6)
        self.assertAlmostEqual(float(closed[3]), 0.5864 * 0.8024 * 0.9487, places=6)
        self.assertAlmostEqual(float(closed[5]), 1.4381 * 1.0843, places=6)

    def test_round_trip_and_followers_are_not_double_counted(self):
        synergy = torch.tensor([[0.2, 0.8], [0.75, 0.1]])
        targets = action_math.inspire_synergy_to_joint_positions(synergy)
        actual = action_math.inspire_joint_positions_to_synergy(targets)
        torch.testing.assert_close(actual, synergy)

        changed_followers = targets.clone()
        changed_followers[:, [2, 3, 5, 7, 9, 11]] = 0.0
        still_master_aggregate = action_math.inspire_joint_positions_to_synergy(changed_followers)
        torch.testing.assert_close(still_master_aggregate, synergy)


if __name__ == "__main__":
    unittest.main()
