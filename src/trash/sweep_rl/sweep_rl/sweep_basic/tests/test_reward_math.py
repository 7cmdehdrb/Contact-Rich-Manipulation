"""Unit tests for simulator-independent single-stage reward kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module(name: str, filename: str):
    path = Path(__file__).parents[1] / "mdp" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_math = _load_module("sweep_basic_reward_math", "reward_math.py")


def test_reaching_reward_increases_near_contact_point():
    score = reward_math.gaussian_distance(
        torch.tensor([0.25, 0.04, 0.00]), std=0.25
    )
    assert score[2] > score[1] > score[0]
    assert score[0] > 0.35
    assert torch.isfinite(score).all()


def test_orientation_score_preserves_sign_and_squares_alignment():
    score = reward_math.signed_square_alignment_score(
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    )
    assert torch.allclose(score, torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0]))


def test_remaining_horizon_failure_charges_only_failed_environments():
    penalty = reward_math.remaining_horizon_failure_kernel(
        torch.tensor([True, False, True]),
        torch.tensor([40, 40, 740]),
        max_episode_length=750,
        minimum_steps=50.0,
    )
    assert torch.allclose(penalty, torch.tensor([710.0, 0.0, 50.0]))


def test_pushing_reward_uses_pose_gate_and_piecewise_y_speed():
    score = reward_math.pushing_target_kernel(
        goal_distance=torch.full((5,), 0.09),
        motion_gate=torch.tensor([False, True, True, True, True]),
        target_y_speed=torch.tensor([0.08, 0.05, 0.051, 0.099, 0.10]),
        goal_threshold=0.03,
        distance_scale=0.18,
        velocity_reward_min=0.05,
        velocity_penalty_min=0.10,
    )
    assert torch.allclose(score, torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0]))
    assert torch.isfinite(score).all()
    initial_contact_score = reward_math.pushing_target_kernel(
        goal_distance=torch.tensor([0.40]),
        motion_gate=torch.tensor([True]),
        target_y_speed=torch.tensor([0.08]),
        goal_threshold=0.03,
        distance_scale=0.18,
        velocity_reward_min=0.05,
        velocity_penalty_min=0.10,
    )
    assert torch.allclose(initial_contact_score, torch.tensor([-0.7222222]))


def test_pushing_reward_near_goal_ignores_pose_gate():
    score = reward_math.pushing_target_kernel(
        goal_distance=torch.tensor([0.02, 0.03]),
        motion_gate=torch.tensor([False, False]),
        target_y_speed=torch.tensor([0.20, 0.20]),
        goal_threshold=0.03,
        distance_scale=0.18,
        velocity_reward_min=0.05,
        velocity_penalty_min=0.10,
    )
    assert torch.allclose(score[0], 2.0 * torch.exp(torch.tensor(-0.10)))
    assert score[1] == 0.0


def test_shelf_clearance_only_penalizes_frames_over_the_shelf():
    positions = torch.tensor(
        [
            [[0.0, 0.0, 1.04], [0.0, 0.0, 1.06], [0.0, 0.0, 1.10]],
            [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]],
        ]
    )
    score = reward_math.shelf_clearance_kernel(
        positions,
        torch.tensor([0.02, 0.02, 0.08]),
        x_bounds=(-0.2, 0.2),
        y_bounds=(-0.5, 0.5),
        surface_height=1.05,
    )
    assert score[0] > 0.0
    assert score[1] == 0.0
    assert torch.isfinite(score).all()
