"""Both push sides keep fingers outward and the palm normal horizontal."""

import math
from itertools import product

import pytest
import torch

from hand_manipulation_test.push_math import push_direction_from_angle, push_path_within_table
from hand_manipulation_test.push_v1_math import push_v1_palm_rotation


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_both_sides_keep_fingers_outward_and_form_proper_rotations(dtype):
    angles = torch.tensor((-10., 0., 10., 170., 180., 190.), dtype=dtype) * math.pi / 180
    direction = push_direction_from_angle(angles)
    rotation = push_v1_palm_rotation(direction)
    thumb_z = direction.new_tensor((1., 1., 1., -1., -1., -1.))
    torch.testing.assert_close(rotation[..., 2, 0], thumb_z)
    torch.testing.assert_close(rotation[..., :2, 0], direction.new_zeros(6, 2))
    torch.testing.assert_close(rotation[..., :, 1], direction)
    torch.testing.assert_close(rotation[..., 2, 1], direction.new_zeros(6))
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3, dtype=dtype).expand(6, -1, -1))
    torch.testing.assert_close(torch.linalg.det(rotation), direction.new_ones(6))
    # H +Z projects world -X onto the plane perpendicular to the command.
    # Both finger axes therefore point from the robot toward the far Table.
    assert bool((rotation[..., 0, 2] < -.98).all())
    torch.testing.assert_close(rotation[..., 2, 2], direction.new_zeros(6))
    assert rotation.dtype == dtype
    assert rotation.device == direction.device


def test_nonunit_direction_is_projected_without_mutating_input():
    direction = torch.tensor([[[0., -3., 2.], [1., 3., -2.]]], dtype=torch.float64)
    before = direction.clone()
    rotation = push_v1_palm_rotation(direction)
    torch.testing.assert_close(direction, before)
    normalized = direction.clone()
    normalized[..., 2] = 0.
    normalized /= normalized.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(rotation[..., :, 1], normalized)
    torch.testing.assert_close(rotation[..., :, 2], direction.new_tensor([[[-1., 0., 0.], [-3./math.sqrt(10.), 1./math.sqrt(10.), 0.]]]))
    assert rotation.shape == (1, 2, 3, 3)


def test_single_direction_has_same_orientation_as_one_row_batch():
    direction = torch.tensor([0., 1., 0.])
    torch.testing.assert_close(push_v1_palm_rotation(direction), push_v1_palm_rotation(direction[None])[0])


@pytest.mark.parametrize("direction", (
    (0., 0., 0.), (0., 0., 1.), (float("nan"), 1., 0.),
    (1., float("inf"), 0.), (1., 0., float("nan")),
))
def test_invalid_direction_is_rejected_before_reset(direction):
    with pytest.raises(ValueError, match="finite with a nonzero horizontal"):
        push_v1_palm_rotation(torch.tensor(direction))


@pytest.mark.parametrize("direction", (torch.tensor(1.), torch.tensor([0., 1.]), torch.tensor([0, 1, 0])))
def test_invalid_direction_contract_is_rejected(direction):
    with pytest.raises(ValueError):
        push_v1_palm_rotation(direction)


@pytest.mark.parametrize("direction", ((1., 0., 0.), (-1., 0., 2.)))
def test_command_parallel_to_outward_axis_is_rejected_instead_of_singular_pose(direction):
    with pytest.raises(ValueError, match="nonzero outward finger projection"):
        push_v1_palm_rotation(torch.tensor(direction))


@pytest.mark.parametrize("origin", ((0., 0.), (-48.75, 37.5)))
def test_far_workspace_contains_the_complete_centered_command_domain(origin):
    # Check every actual sampler endpoint independently of IK, including
    # translated environment coordinates and the full intermediate Cube path.
    corners = list(product((-10., 10., 170., 190.), (.2, .3), (-.002, .002), (-.03, .03)))
    values = torch.tensor(corners, dtype=torch.float64)
    direction = push_direction_from_angle(values[:, 0] * math.pi / 180.)
    centre = values.new_tensor((-.75, 0.)) + values.new_tensor(origin)
    start = torch.stack((centre[0] + .05 - .5 * values[:, 1] * direction[:, 0] + values[:, 2],
                         centre[1] + values[:, 3]), dim=-1)
    goal = start + values[:, 1:2] * direction[:, :2]
    low = centre + values.new_tensor((.02, -.03))
    high = centre + values.new_tensor((.08, .03))
    assert bool(((start >= low) & (start <= high)).all())
    assert bool(push_path_within_table(start, goal, centre, values.new_tensor((.36, 1.)), .06, .01).all())

    # Check intermediate positions against the orientation-safe Cube radius,
    # in addition to the endpoint-based convex-rectangle certificate.
    fractions = torch.linspace(0., 1., 51, dtype=values.dtype)
    path = start[:, None] + fractions[None, :, None] * (goal - start)[:, None]
    extent = values.new_tensor((.36, 1.)) / 2 - math.sqrt(3.) * .06 / 2 - .01
    clearance = extent - (path - centre).abs()
    assert float(clearance.min()) > .0399


def test_common_table_contains_far_paths_and_rejects_previous_near_paths():
    direction = push_direction_from_angle(torch.tensor((10., 170.), dtype=torch.float64) * math.pi / 180.)
    centre = torch.tensor((-.75, 0.), dtype=direction.dtype)
    start = torch.stack((centre[0] + .31 - .15 * direction[:, 0] + .002,
                         direction.new_zeros(2)), dim=-1)
    goal = start + .30 * direction[:, :2]
    assert not bool(push_path_within_table(start, goal, centre, direction.new_tensor((.36, 1.)), .06, .01).any())
    start[:, 0] -= .26
    goal[:, 0] -= .26
    assert bool(push_path_within_table(start, goal, centre, direction.new_tensor((.36, 1.)), .06, .01).all())
