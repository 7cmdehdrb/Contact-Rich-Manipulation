"""Cartesian command round trips using the real Isaac Lab quaternion math."""
import importlib
import math

import pytest
import torch


@pytest.fixture
def manual(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.runtime.wrench_refactor_manual')


@pytest.mark.parametrize('root_yaw', [0., math.pi / 2, -2., None])
def test_world_target_round_trip(manual, tensor_math, root_yaw):
    m = tensor_math
    angles = torch.tensor([[.4, -.3, 2.8], [-.8, .2, -2.]])
    current_q = m.quat_from_euler_xyz(*angles.unbind(-1))
    delta = torch.tensor([[.01, -.02, .005, .03, -.02, .01], [-.02, .01, 0., -.02, .01, -.03]])
    current_p = torch.tensor([[.2, .3, 1.2], [-.3, .1, .8]])
    target_p, target_q = m.apply_delta_pose(current_p, current_q, delta)
    zero = torch.zeros(2)
    root = current_q if root_yaw is None else m.quat_from_euler_xyz(zero, zero, torch.full((2,), root_yaw))
    action = manual.cartesian_action(current_p, current_q, target_p, target_q, root, .03, .15, .1, .5)
    root_q = m.quat_mul(m.quat_conjugate(root), current_q)
    command = action * torch.tensor([.03] * 3 + [.15] * 3)
    dq = m.quat_from_euler_xyz(*command[:, 3:].unbind(-1))
    command[:, 3:] = m.axis_angle_from_quat(dq)
    actual_p, actual_q = m.apply_delta_pose(m.quat_apply_inverse(root, current_p), root_q, command)
    torch.testing.assert_close(m.quat_apply(root, actual_p), target_p, atol=1e-6, rtol=0.)
    assert torch.allclose((m.quat_mul(root, actual_q) * target_q).sum(-1).abs(), torch.ones(2), atol=1e-6)


def test_reference_slew_and_shortest_rotation(manual, tensor_math):
    m = tensor_math
    p = torch.zeros(1, 3)
    q = torch.tensor([[1., 0., 0., 0.]])
    target = torch.tensor([[1., 0., 0.]])
    target_q = -m.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([.2]))
    p1, q1 = manual.advance_reference(p, q, target, target_q, .01, .02)
    torch.testing.assert_close(p1, torch.tensor([[.01, 0., 0.]]))
    torch.testing.assert_close(m.axis_angle_from_quat(q1), torch.tensor([[0., 0., .02]]))
    # The reference keeps progressing even when the actual robot is blocked.
    p2, _ = manual.advance_reference(p1, q1, target, target_q, .01, .02)
    torch.testing.assert_close(p2, torch.tensor([[.02, 0., 0.]]))
    p3, q3 = manual.advance_reference(p, q, p, q, .01, .02)
    torch.testing.assert_close(p3, p)
    torch.testing.assert_close(q3, q)
    assert torch.isfinite(q3).all()


def test_command_norm_caps_and_zero_error(manual):
    p = torch.zeros(1, 3)
    q = torch.tensor([[1., 0., 0., 0.]])
    action = manual.cartesian_action(p, q, torch.ones_like(p), q, q, .03, .15, .03, .15)
    assert float((action[:, :3] * .03).norm()) == pytest.approx(.03)
    assert torch.isfinite(action).all()
    zero = manual.cartesian_action(p, q, p, q, q, .03, .15, .03, .15)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
