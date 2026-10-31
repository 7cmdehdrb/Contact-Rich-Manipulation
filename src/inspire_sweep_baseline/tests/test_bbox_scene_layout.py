"""Independent polygon checks for the torch reset planner, without Isaac Sim."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline/mdp/common/bbox_scene_layout.py'
spec = importlib.util.spec_from_file_location('bbox_scene_layout_tested', PATH)
layout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = layout
spec.loader.exec_module(layout)


def rectangle(center, half, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half @ np.array([[c, s], [-s, c]]) + center


def hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def chain(seq):
        result = []
        for p in seq:
            while len(result) > 1 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result[:-1]
    return np.array(chain(points)+chain(points[::-1]))


def polygon_separated(a, b, margin=0.):
    # Independent vertex projection, not the generator's radius formula.
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0)-poly:
            if np.linalg.norm(edge) < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            pa, pb = a @ normal, b @ normal
            if max(pb.min()-pa.max(), pa.min()-pb.max()) >= margin-2e-6:
                return True
    return False


def polygon_distance(a, b):
    if not polygon_separated(a, b):
        return 0.
    distances = []
    for p, q in ((a, b), (b, a)):
        for start, end in zip(q, np.roll(q, -1, axis=0)):
            delta = end-start
            t = np.clip((p-start) @ delta / (delta @ delta), 0, 1)
            distances.extend(np.linalg.norm(p-start-t[:, None]*delta, axis=1))
    return min(distances)


def inputs(batch, cfg, device='cpu'):
    gen = torch.Generator(device=device).manual_seed(702)
    target = torch.tensor([-.08, .22], device=device).expand(batch, -1).clone()
    goal = target + torch.tensor([.16, .025], device=device)
    th = torch.tensor([.025, .045], device=device).expand(batch, -1).clone()
    ty = torch.linspace(-.8, .8, batch, device=device)
    half = .009 + torch.rand(batch, cfg.max_gray+cfg.max_white, 2, device=device, generator=gen)*.035
    return target, goal, th, ty, half


def assert_geometry(result, args, cfg):
    target, goal, th, ty, half = [x.cpu().numpy() for x in args]
    centers, yaw, active = [x.cpu().numpy() for x in (result.centers, result.yaw, result.active)]
    for row in range(len(target)):
        initial = rectangle(target[row], th[row], ty[row])
        final = rectangle(goal[row], th[row], ty[row])
        down = np.array([0, -(cfg.board[3]-cfg.board[2]+2*th[row].sum())])
        sweep = hull(np.concatenate((initial, final, initial+down, final+down)))
        angle = np.arctan2(*(goal[row]-target[row])[::-1])
        direction = np.array([np.cos(angle), np.sin(angle)])
        rear = rectangle(target[row]-direction*cfg.rear_length/2,
                         np.array([cfg.rear_length, cfg.rear_width])/2, angle)
        polygons = [initial]
        for i in np.flatnonzero(active[row]):
            box = rectangle(centers[row, i], half[row, i], yaw[row, i])
            assert box[:, 0].min() >= cfg.board[0]+cfg.wall_margin-2e-6
            assert box[:, 0].max() <= cfg.board[1]-cfg.wall_margin+2e-6
            assert box[:, 1].min() >= cfg.board[2]+cfg.wall_margin-2e-6
            assert box[:, 1].max() <= cfg.board[3]-cfg.wall_margin+2e-6
            assert polygon_separated(box, sweep, cfg.red_margin)
            interaction = (not polygon_separated(box, sweep, cfg.red_margin+cfg.gray_padding)
                           or (cfg.rear_length > 0 and cfg.rear_width > 0 and not polygon_separated(box, rear)))
            assert interaction == (i < cfg.max_gray)
            for other in polygons:
                assert polygon_distance(box, other) >= cfg.finger_gap-2e-6
            polygons.append(box)
    torch.testing.assert_close(result.actual[:, 0], result.active[:, :cfg.max_gray].sum(-1))
    torch.testing.assert_close(result.actual[:, 1], result.active[:, cfg.max_gray:].sum(-1))
    assert (result.actual <= result.requested).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_heterogeneous_rotated_geometry(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cfg = layout.LayoutConfig(candidates=24, restarts=3, passes=2)
    args = inputs(24, cfg, device)
    result = layout.generate_layout(*args, cfg=cfg, generator=torch.Generator(device=device).manual_seed(9))
    assert_geometry(result, args, cfg)
    assert result.actual.sum() > 100


def test_count_caps_seed_and_zero_requests():
    cfg = layout.LayoutConfig(max_gray=2, max_white=3, restarts=2, passes=1)
    args = inputs(3, cfg)
    requests = torch.tensor([[0, 0], [1, 1], [50, 50]])
    a, b = [layout.generate_layout(*args, cfg=cfg, requested=requests,
            generator=torch.Generator().manual_seed(27)) for _ in range(2)]
    torch.testing.assert_close(a.centers, b.centers)
    torch.testing.assert_close(a.active, b.active)
    assert a.requested.tolist() == [[0, 0], [1, 1], [2, 3]]
    assert not a.active[0].any()
    assert a.complete[0]
    assert_geometry(a, args, cfg)


@pytest.mark.parametrize('empty_batch', [True, False])
def test_empty_dimensions(empty_batch):
    cfg = layout.LayoutConfig(max_gray=0, max_white=0) if not empty_batch else layout.LayoutConfig()
    result = layout.generate_layout(*inputs(0 if empty_batch else 3, cfg), cfg=cfg)
    assert result.active.numel() == 0
    assert result.complete.all()


def test_impossible_objects_do_not_prevent_smaller_later_objects():
    cfg = layout.LayoutConfig(max_gray=0, max_white=3)
    args = inputs(4, cfg)
    args[-1][:, 0] = 2.0  # Impossible in any orientation.
    args[-1][:, 1:] = .015
    result = layout.generate_layout(*args, cfg=cfg, generator=torch.Generator().manual_seed(11))
    assert not result.active[:, 0].any()
    assert result.active[:, 1:].all()  # Known maximum = 2 of these supplied objects.
    assert not result.complete.any()
    assert_geometry(result, args, cfg)


def test_full_board_red_returns_empty_without_relaxation():
    cfg = layout.LayoutConfig(red_margin=2., max_gray=2, max_white=2)
    result = layout.generate_layout(*inputs(2, cfg), cfg=cfg)
    assert not result.active.any()
    assert not result.complete.any()


@pytest.mark.parametrize('motion', [(0., 0.), (.15, 0.), (0., .1), (.1, .07), (-.1, -.1)])
def test_sweep_degeneracies_and_padding_monotonicity(motion):
    cfg = layout.LayoutConfig(rear_length=0.)
    t, g, th, ty, _ = inputs(1, cfg)
    g = t + torch.tensor(motion)
    points = torch.rand(1, 300, 2, generator=torch.Generator().manual_seed(4))
    points[..., 0] -= .5
    points[..., 1] *= .4
    h, y = torch.full((1, 300, 2), .012), torch.linspace(-3., 3., 300)[None]
    red, gray, white = layout.zone_masks(points, h, y, t, g, th, ty, cfg)
    red2, gray2, white2 = layout.zone_masks(points, h, y, t, g, th, ty, replace(cfg, gray_padding=.12))
    assert torch.equal(red, red2)
    assert (~gray | gray2).all()
    assert (~white2 | white).all()
    assert (red.int()+gray.int()+white.int()==1).all()


def test_wrist_projection_and_whole_box_classification():
    cfg = layout.LayoutConfig(rear_length=0.)
    t, g, th, ty, _ = inputs(1, cfg)
    ty.zero_()
    centers = torch.tensor([[[0., .03], [.20, .22], [.40, .30]]])
    half = torch.tensor([[[.02, .02], [.15, .02], [.02, .02]]])
    red, _, white = layout.zone_masks(centers, half, torch.zeros(1, 3), t, g, th, ty, cfg)
    assert red[0, 0]  # Far below path, but wrist access must remain clear.
    assert red[0, 1]  # Centre is outside, footprint intrudes into swept target.
    assert white[0, 2]


def test_gap_boundary_and_offcenter_asset_bounds():
    a = torch.zeros(1, 2)
    h = torch.tensor([[.02, .04]])
    y = torch.zeros(1)
    assert layout.separated(a, h, y, torch.tensor([[.062, 0.]]), h, y, .02)
    assert not layout.separated(a, h, y, torch.tensor([[.059, 0.]]), h, y, .02)
    bounds = torch.tensor([[[-.01, -.02, 0.], [.05, .06, .12]]])
    half, offset = layout.footprint_from_bounds(bounds, torch.tensor([np.pi/2]))
    torch.testing.assert_close(half, torch.tensor([[.03, .04]]))
    torch.testing.assert_close(offset, torch.tensor([[-.02, .02]]))


@pytest.mark.parametrize('changes', [dict(finger_gap=-1), dict(gray_padding=float('nan')),
                                    dict(candidates=0), dict(max_gray=-1), dict(board=(0, 0, 0, 1))])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        layout.LayoutConfig(**changes)


def test_invalid_target_and_counts():
    cfg = layout.LayoutConfig()
    args = inputs(1, cfg)
    with pytest.raises(ValueError, match='integer'):
        layout.generate_layout(*args, cfg=cfg, requested=torch.zeros(1, 2))
    args[1][:, 1] = 1.
    with pytest.raises(ValueError, match='sweep'):
        layout.generate_layout(*args, cfg=cfg)


def test_randomized_commands_requests_and_aspect_ratios():
    cfg = layout.LayoutConfig(candidates=24, restarts=3)
    batch = 256
    rng = torch.Generator().manual_seed(381)
    target = torch.tensor([0., .20]).expand(batch, -1).clone()
    angle = torch.rand(batch, generator=rng) * 6.283185
    goal = target + .08 * torch.stack((angle.cos(), angle.sin()), -1)
    th = .012 + torch.rand(batch, 2, generator=rng) * .025
    ty = torch.rand(batch, generator=rng) * 6.283185
    half = .006 + torch.rand(batch, 9, 2, generator=rng) * .060
    requests = torch.stack((torch.randint(5, (batch,), generator=rng),
                            torch.randint(6, (batch,), generator=rng)), -1)
    args = target, goal, th, ty, half
    originals = [x.clone() for x in args]
    result = layout.generate_layout(*args, cfg=cfg, requested=requests, generator=rng)
    assert_geometry(result, args, cfg)
    for original, current in zip(originals, args):
        torch.testing.assert_close(original, current)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_cuda_classification_agrees():
    cfg = layout.LayoutConfig()
    t, g, th, ty, _ = inputs(4, cfg)
    rng = torch.Generator().manual_seed(77)
    c = torch.rand(4, 100, 2, generator=rng) * torch.tensor([1., .4]) - torch.tensor([.5, 0.])
    h = .005 + .05 * torch.rand(4, 100, 2, generator=rng)
    y = 6.28 * torch.rand(4, 100, generator=rng)
    cpu = layout.zone_masks(c, h, y, t, g, th, ty, cfg)
    gpu = layout.zone_masks(*(x.cuda() for x in (c, h, y, t, g, th, ty)), cfg)
    for a, b in zip(cpu, gpu):
        assert torch.equal(a, b.cpu())
