from __future__ import annotations

import math

import pytest
import torch

from hand_manipulation_rl.mdp.approach_rewards import (
    contact_path_potential,
    one_sided_height_safety_penalty,
    potential_progress_rate,
    safe_surface_approach_reward,
)


def _reward(points: torch.Tensor, *, safe_height: float | torch.Tensor = 0.075) -> torch.Tensor:
    objects = torch.zeros_like(points)
    directions = torch.zeros_like(points)
    directions[:, 1] = 1.0
    return safe_surface_approach_reward(
        points,
        objects,
        directions,
        safe_height,
        object_half_extent_m=0.030,
        planar_capture_radius_m=0.015,
        planar_sigma_m=0.080,
        height_sigma_m=0.030,
    )


def test_approach_reward_increases_toward_upstream_object_face() -> None:
    points = torch.tensor(
        [
            [0.0, -0.140, 0.075],
            [0.0, -0.080, 0.075],
            [0.0, -0.030, 0.075],
        ]
    )
    reward = _reward(points)
    assert reward[0] < reward[1] < reward[2]
    assert reward[2].item() == pytest.approx(1.0)


def test_capture_radius_removes_penetration_incentive() -> None:
    near_face = _reward(torch.tensor([[0.0, -0.030, 0.075]])).item()
    outside_capture = _reward(torch.tensor([[0.0, -0.040, 0.075]])).item()
    inside_capture = _reward(torch.tensor([[0.0, -0.020, 0.075]])).item()
    past_face = _reward(torch.tensor([[0.0, 0.000, 0.075]])).item()
    assert near_face == pytest.approx(1.0)
    assert outside_capture == pytest.approx(1.0)
    assert past_face < inside_capture < near_face


def test_descending_from_safe_height_reduces_reward() -> None:
    safe = _reward(torch.tensor([[0.0, -0.030, 0.075]])).item()
    descended = _reward(torch.tensor([[0.0, -0.030, 0.025]])).item()
    assert descended == pytest.approx(safe * math.exp(-0.050 / 0.030))
    assert descended < safe


def test_v1_progress_is_signed_and_zero_for_no_motion() -> None:
    previous = torch.tensor([0.25, 0.25, 0.25])
    current = torch.tensor([0.35, 0.25, 0.10])
    rate = potential_progress_rate(current, previous, 0.02)
    torch.testing.assert_close(rate, torch.tensor([5.0, 0.0, -7.5]))
    # RewardManager's integration recovers the exact telescoping delta.
    torch.testing.assert_close(rate * 0.02, current - previous)


def test_v1_contact_path_caps_hand_only_progress_and_unlocks_with_object_motion() -> None:
    initial_hand = torch.zeros(5, 3)
    current_hand = initial_hand.clone()
    current_hand[:, 1] = torch.tensor([0.050, 0.105, 0.200, 0.200, 0.400])
    initial_object = torch.zeros_like(initial_hand)
    current_object = initial_object.clone()
    current_object[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.060, 0.300])
    direction = torch.zeros_like(initial_hand)
    direction[:, 1] = 1.0
    command_distance = torch.full((5,), 0.180)
    target_height = torch.zeros(5)

    potential = contact_path_potential(
        current_hand,
        initial_hand,
        current_object,
        initial_object,
        direction,
        command_distance,
        target_height,
        free_approach_distance_m=0.105,
    )
    approach_share = 0.35
    push_share = 1.0 - approach_share
    torch.testing.assert_close(
        potential,
        torch.tensor(
            [
                approach_share * 0.050 / 0.105,
                approach_share,
                approach_share,
                approach_share + push_share * 0.060 / 0.180,
                1.0,
            ]
        ),
    )


def test_v1_contact_path_couples_calibrated_descent_to_forward_progress() -> None:
    initial_hand = torch.tensor([[0.0, 0.0, 1.0]])
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    kwargs = {
        "initial_surface_position_w": initial_hand,
        "object_position_w": torch.zeros(1, 3),
        "initial_object_position_w": torch.zeros(1, 3),
        "command_direction_w": direction,
        "command_distance_m": torch.tensor([0.180]),
        "target_height_w": torch.tensor([0.980]),
        "free_approach_distance_m": 0.105,
        "height_weight": 1.5,
    }
    reset = contact_path_potential(initial_hand, **kwargs)
    down_only = contact_path_potential(
        torch.tensor([[0.0, 0.0, 0.980]]), **kwargs
    )
    half_path = contact_path_potential(
        torch.tensor([[0.0, 0.0525, 0.990]]), **kwargs
    )
    half_forward_wrong_height = contact_path_potential(
        torch.tensor([[0.0, 0.0525, 1.000]]), **kwargs
    )
    assert reset.item() == pytest.approx(0.0)
    assert down_only.item() < reset.item()
    assert half_path.item() > half_forward_wrong_height.item() > reset.item()


def test_v1_contact_path_accepts_branch_specific_free_distances() -> None:
    initial = torch.zeros(2, 3)
    current = initial.clone()
    current[:, 1] = torch.tensor([0.080, 0.080])
    direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    free_distance = torch.tensor([0.055, 0.105])
    value = contact_path_potential(
        current,
        initial,
        torch.zeros_like(initial),
        torch.zeros_like(initial),
        direction,
        torch.full((2,), 0.180),
        torch.zeros(2),
        free_approach_distance_m=free_distance,
    )
    torch.testing.assert_close(
        value,
        torch.tensor([0.35, 0.35 * 0.080 / 0.105]),
    )


def test_v1_height_safety_is_one_sided() -> None:
    initial = torch.tensor([0.075, 0.075, 0.075, 0.075])
    current = torch.tensor([0.080, 0.073, 0.062, 0.040])
    penalty = one_sided_height_safety_penalty(
        current,
        initial,
        tolerance_m=0.003,
        band_m=0.025,
    )
    torch.testing.assert_close(
        penalty,
        torch.tensor([0.0, 0.0, -0.16, -1.0]),
    )


def test_reward_is_translation_invariant_and_accepts_per_row_heights() -> None:
    points = torch.tensor([[0.0, -0.10, 0.065], [0.0, -0.10, 0.112]])
    objects = torch.zeros_like(points)
    directions = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    heights = torch.tensor([0.065, 0.112])
    kwargs = {
        "object_half_extent_m": 0.030,
        "planar_capture_radius_m": 0.015,
        "planar_sigma_m": 0.080,
        "height_sigma_m": 0.030,
    }
    baseline = safe_surface_approach_reward(points, objects, directions, heights, **kwargs)
    translation = torch.tensor([1.0, -2.0, 0.5])
    translated = safe_surface_approach_reward(
        points + translation,
        objects + translation,
        directions,
        heights,
        **kwargs,
    )
    torch.testing.assert_close(baseline, translated)
    torch.testing.assert_close(baseline[0], baseline[1])


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("object_half_extent_m", 0.0),
        ("planar_capture_radius_m", -0.01),
        ("planar_sigma_m", 0.0),
        ("height_sigma_m", 0.0),
    ),
)
def test_invalid_reward_scales_are_rejected(keyword: str, value: float) -> None:
    point = torch.zeros(1, 3)
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    kwargs = {
        "object_half_extent_m": 0.030,
        "planar_capture_radius_m": 0.015,
        "planar_sigma_m": 0.080,
        "height_sigma_m": 0.030,
    }
    kwargs[keyword] = value
    with pytest.raises(ValueError):
        safe_surface_approach_reward(point, point, direction, 0.075, **kwargs)


def test_invalid_shapes_and_zero_direction_are_rejected() -> None:
    point = torch.zeros(1, 3)
    direction = torch.tensor([[0.0, 1.0, 0.0]])
    kwargs = {
        "object_half_extent_m": 0.030,
        "planar_capture_radius_m": 0.015,
        "planar_sigma_m": 0.080,
        "height_sigma_m": 0.030,
    }
    with pytest.raises(ValueError, match="identical"):
        safe_surface_approach_reward(point, torch.zeros(2, 3), direction, 0.075, **kwargs)
    with pytest.raises(ValueError, match="non-zero planar"):
        safe_surface_approach_reward(point, point, torch.zeros_like(point), 0.075, **kwargs)
