from __future__ import annotations

import math

import pytest
import torch

from hand_manipulation_rl.mdp.rewards import (
    action_change_reward,
    contact_normal_alignment_reward,
    goal_reward,
    height_increase,
    quadratic_risk_ramp,
    resultant_contact_normal,
    risk_penalty,
    sum_link_normal_force,
    tactile_contact_reward,
    time_penalty_rate,
    upright_tilt_angle,
)


def test_goal_reward_uses_3d_distance() -> None:
    object_position = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    goal = torch.zeros_like(object_position)
    reward = goal_reward(object_position, goal, sigma_position=1.0)
    assert torch.allclose(reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_resultant_contact_normal_and_alignment_contract() -> None:
    normals = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    loads = torch.tensor([[2.0, 0.0]])
    resultant, valid = resultant_contact_normal(normals, loads)
    assert torch.allclose(resultant, torch.tensor([[1.0, 0.0, 0.0]]))
    assert valid.item()

    command = torch.tensor([[1.0, 0.0, 0.0]])
    assert contact_normal_alignment_reward(resultant, command, valid_contact=valid).item() == 0.0
    opposite = contact_normal_alignment_reward(-resultant, command, valid_contact=valid)
    assert opposite.item() == -1.0
    absent = contact_normal_alignment_reward(
        torch.zeros_like(resultant), command, valid_contact=torch.tensor([False])
    )
    assert absent.item() == 0.0


def test_tactile_reward_uses_only_17_selected_bits() -> None:
    none = torch.zeros(1, 17)
    full = torch.ones(1, 17)
    one = torch.zeros(1, 17)
    one[0, 0] = 1
    reward = tactile_contact_reward(torch.cat((none, full, one)), beta=0.5)
    assert torch.allclose(reward, torch.tensor([-1.0, 0.0, -0.5 + 0.5 / 17.0]))
    with pytest.raises(ValueError, match="17"):
        tactile_contact_reward(torch.ones(1, 18))


def test_boolean_tactile_preserves_nonuniform_region_weights() -> None:
    tactile = torch.zeros(1, 17, dtype=torch.bool)
    tactile[0, 0] = True
    weights = torch.tensor([2.0] + [1.0] * 16)
    reward = tactile_contact_reward(tactile, beta=0.0, region_weights=weights)
    assert torch.allclose(reward, torch.tensor([2.0 / 18.0 - 1.0]))


def test_first_action_has_no_change_penalty() -> None:
    current = torch.ones(2, 8)
    previous = torch.zeros_like(current)
    reward = action_change_reward(
        current,
        previous,
        previous_action_valid=torch.tensor([False, True]),
        arm_coefficient=2.0,
        hand_coefficient=3.0,
    )
    assert torch.allclose(reward, torch.tensor([0.0, -18.0]))


def test_risk_ramp_and_time_rate_have_expected_sign_and_scale() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(
        quadratic_risk_ramp(values, 1.0, 3.0), torch.tensor([0.0, 0.0, 0.25, 1.0])
    )
    assert torch.allclose(
        risk_penalty(values, 1.0, 3.0), torch.tensor([0.0, 0.0, -0.25, -1.0])
    )
    assert time_penalty_rate(4.0) == -0.25
    with pytest.raises(ValueError, match="greater"):
        quadratic_risk_ramp(values, 1.0, 1.0)


def test_tilt_height_and_board_force_helpers() -> None:
    tilt = upright_tilt_angle(
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert torch.allclose(tilt, torch.tensor([0.0, math.pi / 2]))
    assert torch.allclose(
        height_increase(torch.tensor([0.3]), torch.tensor([0.2])), torch.tensor([0.1])
    )
    forces = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    assert sum_link_normal_force(forces).item() == 2.0
