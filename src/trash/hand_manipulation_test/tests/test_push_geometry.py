"""Physical side-pose and complete Cube-path contracts without a simulator."""

import math

import torch

from hand_manipulation_test.push_math import (
    cube_circumsphere_radius,
    cube_planar_support_radius,
    push_direction_from_angle,
    push_palm_start_pose,
    push_path_within_table,
    push_start_xy_bounds,
)


def _angles(dtype=torch.float64):
    return torch.tensor([-10., 0., 10., 170., 180., 190.], dtype=dtype) * math.pi / 180.0


def test_right_left_convention_preserves_horizontal_palm_and_outward_fingers():
    direction = push_direction_from_angle(_angles())
    cube = direction.new_tensor([-.45, 0., 1.08]).expand(6, -1)
    control, rotation, palm = push_palm_start_pose(
        cube, direction, cube.new_tensor([.14, 0., .10]),
        cube.new_tensor([.00045, .047734, .12023]), cube.new_tensor([0., .05, .1]),
    )
    torch.testing.assert_close(direction[1], direction.new_tensor([0., 1., 0.]), atol=1e-14, rtol=0.)
    torch.testing.assert_close(direction[4], direction.new_tensor([0., -1., 0.]), atol=1e-14, rtol=0.)
    preferred = direction.new_tensor([-1., 0., 0.]).expand(6, -1)
    expected_finger = torch.nn.functional.normalize(preferred - (preferred * direction).sum(-1, keepdim=True) * direction, dim=-1)
    torch.testing.assert_close(rotation[:, :, 2], expected_finger)
    torch.testing.assert_close(rotation[:3, :, 0], direction.new_tensor([0., 0., 1.]).expand(3, -1))
    torch.testing.assert_close(rotation[3:, :, 0], direction.new_tensor([0., 0., -1.]).expand(3, -1))
    torch.testing.assert_close(rotation[:, :, 1], direction)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3, dtype=cube.dtype).expand(6, -1, -1))
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(6, dtype=cube.dtype))
    assert bool((rotation[:, 0, 2] < -.98).all())
    torch.testing.assert_close(palm[:, 2] - cube[:, 2], cube.new_full((6,), .10))
    assert bool(torch.isfinite(control).all())


def test_exposed_face_offset_and_env_origin_are_applied_exactly_once():
    direction = push_direction_from_angle(_angles())
    cube = direction.new_tensor([-.45, -.02, 1.08]).expand(6, -1)
    palm_h = cube.new_tensor([.00045, .047734, .12023])
    control_h = cube.new_tensor([0., .05, .1])
    offset = cube.new_tensor([.14, -.004, .03])
    control, rotation, palm = push_palm_start_pose(cube, direction, offset, palm_h, control_h)
    actual_palm = control + (rotation @ (palm_h - control_h).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual_palm, palm)
    translation = cube.new_tensor([5., -2.5, 0.])
    shifted_control, shifted_rotation, shifted_palm = push_palm_start_pose(cube + translation, direction, offset, palm_h, control_h)
    torch.testing.assert_close(shifted_control, control + translation)
    torch.testing.assert_close(shifted_palm, palm + translation)
    torch.testing.assert_close(shifted_rotation, rotation)


def test_approved_cube_box_contains_all_orientation_safe_30cm_paths():
    angles = _angles()
    direction = push_direction_from_angle(angles)
    corners = angles.new_tensor([[-.71, -.06], [-.71, .06], [-.69, -.06], [-.69, .06]])
    starts = corners[:, None].expand(4, 6, 2)
    goals = starts + .30 * direction[None, :, :2]
    center, size = angles.new_tensor([-.75, 0.]), angles.new_tensor([.36, 1.])
    assert bool(push_path_within_table(starts, goals, center, size, .06, .01).all())
    # Test interior segment points too, including footprint points at the
    # worst arbitrary 3-D Cube orientation rather than just its centre.
    radius = cube_circumsphere_radius(.06) + .01
    for fraction in (0., .25, .5, .75, 1.):
        point = starts + fraction * (goals - starts)
        assert bool((point >= center - size / 2 + radius).all())
        assert bool((point <= center + size / 2 - radius).all())


def test_command_bounds_reject_long_paths_and_preserve_endpoint_clearance():
    dtype = torch.float64
    direction = push_direction_from_angle(_angles(dtype))[:, :2]
    center = torch.tensor([-.5, 0.], dtype=dtype)
    size = torch.tensor([.36, 1.], dtype=dtype)
    low, high = push_start_xy_bounds(
        center - size / 2, center + size / 2, center, size, .3 * direction, .06, .01,
    )
    assert bool((low <= high).all())
    assert bool(push_path_within_table(low, low + .3 * direction, center, size, .06, .01).all())
    assert bool(push_path_within_table(high, high + .3 * direction, center, size, .06, .01).all())
    invalid_low, invalid_high = push_start_xy_bounds(
        center - size / 2, center + size / 2, center, size, 2. * direction, .06, .01,
    )
    assert bool((invalid_low > invalid_high).any(dim=-1).all())


def test_cube_contact_plane_is_behind_cube_not_at_its_centre():
    direction = push_direction_from_angle(_angles())
    support = cube_planar_support_radius(direction, .06)
    torch.testing.assert_close(support[[1, 4]], support.new_full((2,), .03))
    assert bool((support[[0, 2, 3, 5]] > .034).all())
    cube = direction.new_tensor([-.45, 0., 1.08]).expand(6, -1)
    face = cube - support[:, None] * direction
    torch.testing.assert_close(((cube - face) * direction).sum(-1), support)
    # Every physical Cube vertex lies on or ahead of the supporting plane.
    corners = direction.new_tensor([[x, y] for x in (-.03, .03) for y in (-.03, .03)])
    projections = (corners[None] * direction[:, None, :2]).sum(-1) + support[:, None]
    assert bool((projections >= -1e-14).all())
