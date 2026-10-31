"""Final shelf/root/slot integration around the independent OBB polygon checks."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from test_bbox_scene_layout import assert_geometry, layout


@pytest.fixture
def adapter(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.final_obb_geometry')


def bounds_batch(n, device='cpu'):
    rng = torch.Generator(device=device).manual_seed(81)
    half = .01 + .025*torch.rand(n, 21, 2, device=device, generator=rng)
    offset = (torch.rand(n, 21, 2, device=device, generator=rng)-.5)*.015
    result = torch.zeros(n, 21, 2, 3, device=device)
    result[..., 0, :2], result[..., 1, :2] = offset-half, offset+half
    result[..., 0, 2] = -.013
    result[..., 1, 2] = .080
    return result


def assert_shelf_geometry(bounds, local, yaw, active, delta, actual, padding_fraction=0.):
    """Check physical root poses after shelf->board conversion, independently of adapter."""
    indices = [1, 2, 3, 4, 11, 12, 13, 14, 15]
    half = (bounds[..., 1, :2]-bounds[..., 0, :2])/2
    offset = (bounds[..., 1, :2]+bounds[..., 0, :2])/2
    c, s = yaw.cos(), yaw.sin()
    rotated = torch.stack((c*offset[..., 0]-s*offset[..., 1], s*offset[..., 0]+c*offset[..., 1]), -1)
    center = local[..., :2] + rotated
    board = torch.stack((center[..., 1], .2-center[..., 0]), -1)
    target = board[:, 0]
    goal = target + torch.stack((delta[:, 1], -delta[:, 0]), -1)
    requests = torch.tensor([4, 5], device=bounds.device).expand(len(bounds), -1)
    cfg = layout.LayoutConfig(gray_padding=.06*(1+padding_fraction))
    output = SimpleNamespace(centers=board[:, indices], yaw=yaw[:, indices]-torch.pi/2,
                             active=active[:, indices], actual=actual, requested=requests)
    assert_geometry(output, (target, goal, half[:, 0], yaw[:, 0]-torch.pi/2, half[:, indices]), cfg)
    assert not active[:, [5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20]].any()
    assert torch.all(local[..., 2][~active] == 100.)
    torch.testing.assert_close((local[..., 2]+bounds[..., 0, 2])[active],
                               torch.full_like(local[..., 2][active], 1.05))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('padding', [0., .15])
def test_shelf_offset_yaw_height_and_capped_slots(adapter, device, padding):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    bounds = bounds_batch(32, device)
    result = adapter.sample_shelf_layout(bounds, torch.Generator(device=device).manual_seed(17),
        gray_obstacles=99, white_obstacles=99, gray_red_padding_fraction=padding)
    assert_shelf_geometry(bounds, *result, padding_fraction=padding)
    local, yaw, _, delta, actual = result
    assert (torch.linalg.vector_norm(local[:, 0, :2], dim=-1) <= .020001).all()
    assert (delta[:, 1].abs()/delta.norm(dim=-1) >= .5-1e-5).all()
    assert (delta.norm(dim=-1) >= .05-1e-5).all()
    assert (delta.norm(dim=-1) <= .15+1e-5).all()
    assert (yaw[:, 0] == 0).all()
    assert (actual <= torch.tensor([4, 5], device=device)).all()


def test_reset_seed_and_failed_slot_identity(adapter):
    bounds = bounds_batch(3)
    bounds[:, 1, 0, :2] = -2
    bounds[:, 1, 1, :2] = 2
    outputs = [adapter.sample_shelf_layout(bounds, torch.Generator().manual_seed(8),
               gray_obstacles=4, white_obstacles=5) for _ in range(2)]
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b)
    local, yaw, active, delta, counts = outputs[0]
    assert not active[:, 1].any()
    assert active[:, 2:5].any()
    assert_shelf_geometry(bounds, local, yaw, active, delta, counts)


def test_impossible_target_fails_explicitly(adapter):
    bounds = bounds_batch(1)
    bounds[:, 0, 0, :2], bounds[:, 0, 1, :2] = -1, 1
    with pytest.raises(ValueError, match='No feasible'):
        adapter.sample_shelf_layout(bounds, torch.Generator().manual_seed(1))
