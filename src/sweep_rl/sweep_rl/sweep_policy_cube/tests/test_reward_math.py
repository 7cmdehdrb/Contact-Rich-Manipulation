"""Boundary tests for the copied fixed Sweep-Policy reward kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "mdp" / "reward_math.py"
SPEC = importlib.util.spec_from_file_location("sweep_policy_cube_reward_math", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reward_math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward_math)


def test_axis_alignment_error_uses_shortest_directed_angle():
    alignment = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected = torch.acos(alignment)
    torch.testing.assert_close(
        reward_math.axis_alignment_error_kernel(alignment), expected
    )
    assert expected[-1] == 0.0
    torch.testing.assert_close(expected[2], torch.tensor(torch.pi / 2))
    torch.testing.assert_close(expected[0], torch.tensor(torch.pi))


def test_axis_alignment_error_clamps_roundoff_outside_dot_product_range():
    alignment = torch.tensor([-1.000001, 1.000001])
    error = reward_math.axis_alignment_error_kernel(alignment)
    torch.testing.assert_close(error, torch.tensor([torch.pi, 0.0]))
    assert torch.isfinite(error).all()


def test_ee_y_shelf_z_alignment_uses_rotation_matrix_columns():
    # +90 degrees about world x maps the EE local y-axis onto world/shelf +z.
    ee_rotation = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]
    )
    shelf_rotation = torch.eye(3).unsqueeze(0)
    alignment = reward_math.ee_y_shelf_z_alignment_kernel(
        ee_rotation, shelf_rotation
    )
    torch.testing.assert_close(alignment, torch.ones(1))
    torch.testing.assert_close(
        reward_math.axis_alignment_error_kernel(alignment), torch.zeros(1)
    )


def test_push_success_branch_has_priority_below_strict_threshold():
    reward = reward_math.pushing_reward_kernel(
        distance=torch.tensor([0.029, 0.030]),
        contact_distance=torch.tensor([1.0, 1.0]),
        wrist_y_error=torch.tensor([1.0, 1.0]),
        target_y_velocity=torch.tensor([1.0, 1.0]),
    )
    assert reward[0] > 0.0
    assert reward[1] == 0.0


def test_push_gate_and_velocity_boundaries_match_original_formula():
    reward = reward_math.pushing_reward_kernel(
        distance=torch.full((4,), 0.10),
        contact_distance=torch.tensor([0.039, 0.040, 0.039, 0.039]),
        wrist_y_error=torch.tensor([0.039, 0.039, 0.040, 0.039]),
        target_y_velocity=torch.tensor([0.051, 0.051, 0.051, 0.10]),
    )
    base = 1.0 - 0.10 / 0.18
    torch.testing.assert_close(
        reward, torch.tensor([base + 0.5, 0.0, 0.0, base - 0.5])
    )


def test_fixed_policy_hand_target_matches_reaching_bench_offsets():
    cube_position = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    result = reward_math.fixed_policy_hand_target(
        cube_position,
        0.06,
        torch.tensor([0.18, -0.18, 0.0]),
    )
    torch.testing.assert_close(
        result,
        torch.tensor(
            [[1.00, 1.94, 3.02], [1.00, 2.06, 3.02], [1.00, 2.00, 3.02]]
        ),
    )
    torch.testing.assert_close(
        cube_position,
        torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        ),
    )


def test_reach_fine_position_kernel_matches_isaac_lab_boundaries():
    distance = torch.tensor([0.0, 0.1, 1.0])
    reward = reward_math.position_error_tanh_kernel(distance, std=0.1)
    expected = 1.0 - torch.tanh(distance / 0.1)
    torch.testing.assert_close(reward, expected)
    assert reward[0] == 1.0
    assert reward[0] > reward[1] > reward[2]
