from __future__ import annotations

import math

import pytest
import torch

from hand_manipulation_rl.mdp.approach_rewards import safe_surface_approach_reward


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
