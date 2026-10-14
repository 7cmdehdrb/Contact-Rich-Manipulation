"""Physical contact reset geometry, independent of the simulator runtime."""

import torch

from hand_manipulation_test.action_math import quaternion_from_rotation_vector, quaternion_to_matrix
from hand_manipulation_test.contact_math import oriented_box_overlap, palm_facing_rotation, palm_facing_start_pose


def test_strict_facing_uses_exposed_palm_face_and_keeps_its_nominal_position():
    cube = torch.tensor([[-.7, -.18, 1.08], [-.7, .18, 1.08]], dtype=torch.float64)
    signs = torch.tensor([-1.0, 1.0], dtype=cube.dtype)
    offset = cube.new_tensor([[.14, .002, .1], [.14, -.003, .1]])
    palm_h = cube.new_tensor([.00045, .04772, .12023])
    c_h = cube.new_tensor([0., .05, .1])
    c_pos, rotation, palm_pos = palm_facing_start_pose(cube, signs, offset, palm_h, c_h)
    actual_palm = c_pos + (rotation @ (palm_h - c_h).expand(2, -1).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual_palm, palm_pos)
    to_cube = torch.nn.functional.normalize(cube - actual_palm, dim=-1)
    torch.testing.assert_close(rotation[:, :, 1], to_cube)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3, dtype=cube.dtype).expand(2, -1, -1))
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2, dtype=cube.dtype))
    # The original planar normal would miss the strictly 3-D target direction.
    assert bool((rotation[:, 2, 1] < -.4).all())
    assert bool((rotation[:, 0, 2] < -.9).all())


def test_fixed_cube_range_has_an_eligible_side_at_all_four_corners():
    cube = torch.tensor([[-.82, -.22, 1.08], [-.82, .22, 1.08], [-.63, -.22, 1.08], [-.63, .22, 1.08]])
    offset = cube.new_tensor([.14, 0., .1]).expand(4, -1)
    palm = cube.new_tensor([.00045, .04772, .12023])
    control = cube.new_tensor([0., .05, .1])
    eligible = []
    for sign in (-1., 1.):
        c_pos, _, _ = palm_facing_start_pose(cube, cube.new_full((4,), sign), offset, palm, control)
        eligible.append((c_pos[:, 1] >= -.25) & (c_pos[:, 1] <= .25))
    assert bool((eligible[0] | eligible[1]).all())
    assert not bool((eligible[0] & eligible[1]).any())


def test_palm_pose_sampling_is_translation_equivariant():
    cube = torch.tensor([[-.64, .03, 1.08]], dtype=torch.float64)
    inputs = (cube.new_tensor([1.]), cube.new_tensor([[.14, -.004, .103]]), cube.new_tensor([.00045, .04772, .12023]), cube.new_tensor([0., .05, .1]))
    c, rotation, palm = palm_facing_start_pose(cube, *inputs)
    translation = cube.new_tensor([[2.5, -5., 0.]])
    shifted_c, shifted_rotation, shifted_palm = palm_facing_start_pose(cube + translation, *inputs)
    torch.testing.assert_close(shifted_c, c + translation)
    torch.testing.assert_close(shifted_palm, palm + translation)
    torch.testing.assert_close(shifted_rotation, rotation)


def test_preferred_finger_axis_has_a_finite_parallel_normal_fallback():
    normal = torch.tensor([[-1., 0., 0.], [1., 0., 0.], [0., 0., -1.]])
    rotation = palm_facing_rotation(normal)
    assert bool(torch.isfinite(rotation).all())
    torch.testing.assert_close(rotation[:, :, 1], normal)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(3))


def test_obb_gate_treats_touch_as_contact_and_respects_outward_pad_extent():
    identity = torch.eye(3).expand(3, -1, -1)
    center = torch.zeros(3, 3)
    other = torch.tensor([[.06, 0., 0.], [.0601, 0., 0.], [.065, 0., 0.]])
    extent = torch.full((3, 3), .03)
    overlap = oriented_box_overlap(center, identity, extent, other, identity, extent)
    assert overlap.tolist() == [True, False, False]
    padded = oriented_box_overlap(center, identity, extent + .004, other, identity, extent)
    assert padded.tolist() == [True, True, False]


def test_obb_gate_is_symmetric_and_invariant_under_global_rotation():
    dtype = torch.float64
    a = torch.tensor([[0., 0., 0.], [.2, .05, .1]], dtype=dtype)
    b = torch.tensor([[.01, .01, .01], [.4, .25, .25]], dtype=dtype)
    ra = quaternion_to_matrix(quaternion_from_rotation_vector(torch.tensor([[.3, .1, -.2], [1., .2, .4]], dtype=dtype)))
    rb = quaternion_to_matrix(quaternion_from_rotation_vector(torch.tensor([[.1, -.5, .6], [.2, .8, -.3]], dtype=dtype)))
    ea = torch.tensor([[.03, .04, .06], [.04, .02, .07]], dtype=dtype)
    eb = torch.full((2, 3), .03, dtype=dtype)
    result = oriented_box_overlap(a, ra, ea, b, rb, eb)
    assert result.tolist() == [True, False]
    assert torch.equal(result, oriented_box_overlap(b, rb, eb, a, ra, ea))
    global_rotation = quaternion_to_matrix(quaternion_from_rotation_vector(torch.tensor([.7, -.3, .5], dtype=dtype)))
    translation = torch.tensor([1., -2., .4], dtype=dtype)
    shifted = oriented_box_overlap(
        a @ global_rotation.T + translation, global_rotation @ ra, ea,
        b @ global_rotation.T + translation, global_rotation @ rb, eb,
    )
    assert torch.equal(result, shifted)
