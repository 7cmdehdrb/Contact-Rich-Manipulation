"""Geometric invariants for the independent Revision reset adapter."""

import importlib
import math
import torch


def bounds(count):
    half = torch.linspace(.025, .055, 20).reshape(10, 2)
    values = torch.zeros(count, 10, 2, 3)
    values[:, :, 0, :2] = -half
    values[:, :, 1, :2] = half
    values[:, :, 1, 2] = .12
    # Asymmetric origins must be converted to footprint centres and back.
    values[:, :, :, 0] += .01
    values[:, :, :, 2] -= .025
    return values


def test_placement_bounds_gap_and_bottom(load_mdp):
    layout = importlib.import_module('_baseline_cpu_pkg.mdp.common.revision_layout')
    obb = importlib.import_module('_baseline_cpu_pkg.mdp.common.bbox_scene_layout')
    b = bounds(32)
    local, yaw, active, delta, result = layout.revision_layout(b, torch.Generator().manual_seed(12), 4, 5)
    assert torch.allclose(local[..., 2][active] + b[..., 0, 2][active], torch.full_like(local[..., 2][active], 1.05))
    assert torch.all(delta.norm(dim=-1) <= .150001)
    assert torch.all(delta.norm(dim=-1) >= .05)
    half, offset = obb.footprint_from_bounds(b, yaw)
    centers = local[..., :2] + offset
    # Independent four-corner boundary check in the shelf coordinate system.
    signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    corners = torch.einsum('bmki,bmij->bmkj', half[:, :, None] * signs, obb.axes(yaw)) + centers[:, :, None]
    assert (corners[..., 0].abs()[active] <= .195001).all()
    assert (corners[..., 1].abs()[active] <= .495001).all()
    for a in range(10):
        for z in range(a + 1, 10):
            both = active[:, a] & active[:, z]
            assert obb.separated(centers[:, a], half[:, a], yaw[:, a], centers[:, z], half[:, z], yaw[:, z], .02 - 1e-6)[both].all()
    assert (result.actual[:, 0] <= 4).all() and (result.actual[:, 1] <= 5).all()
    assert (local[..., 2][~active] >= 50).all()


def test_layout_repeatability_and_zero_obstacles(load_mdp):
    layout = importlib.import_module('_baseline_cpu_pkg.mdp.common.revision_layout')
    b = bounds(8)
    a = layout.revision_layout(b, torch.Generator().manual_seed(92), 0, 0)
    z = layout.revision_layout(b, torch.Generator().manual_seed(92), 0, 0)
    for i in range(4):
        torch.testing.assert_close(a[i], z[i])
    assert a[2][:, 0].all() and not a[2][:, 1:].any()
    assert a[-1].complete.all()
    assert torch.all(a[1] == 0)


def test_direction_deadzone(load_mdp):
    layout = importlib.import_module('_baseline_cpu_pkg.mdp.common.revision_layout')
    target, goal = layout.sample_revision_commands(bounds(2048)[:, 0], torch.Generator().manual_seed(3), excluded_x_half_angle=math.pi / 6)
    delta = goal - target
    assert (target.norm(dim=-1) <= .020001).all()
    assert ((delta[:, 1].abs() / delta.norm(dim=-1)) >= .5 - 1e-6).all()
    assert (delta[:, 0] > 0).any() and (delta[:, 0] < 0).any()
