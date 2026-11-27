"""Independent geometry checks for the bounded initial-pose selector."""
import importlib
import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def initial_pose(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_initial_pose')


def test_red_samples_exclude_whole_sweep_lane_and_reproduce(initial_pose):
    start = torch.tensor([[0., 0.], [.01, -.01], [0., 0.]])
    goal = torch.tensor([[0., .12], [-.1, -.02], [.1, 0.]])
    radius = torch.tensor([.08, .08, .08])
    points, valid = initial_pose.red_candidates(start, goal, radius, torch.Generator().manual_seed(42), 2048)
    again = initial_pose.red_candidates(start, goal, radius, torch.Generator().manual_seed(42), 2048)
    assert torch.equal(points, again[0]) and torch.equal(valid, again[1])
    # Independent closest-point scalar oracle, including segment endpoints.
    for row in range(3):
        a, b = start[row].tolist(), goal[row].tolist()
        for x, y in points[row, valid[row]].tolist():
            dx, dy = b[0]-a[0], b[1]-a[1]
            t = min(1., max(0., ((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
            assert math.hypot(x-a[0]-t*dx, y-a[1]-t*dy) >= .08-1e-7
            assert abs(x)<.195 and abs(y)<.495
    assert valid[:2].any(-1).all()


def test_segment_endpoints_and_zero_length(initial_pose):
    points = torch.tensor([[[-1., 0.], [.5, .2], [2., 0.]]])
    actual = initial_pose.segment_distance(points, torch.tensor([[0., 0.]]), torch.tensor([[1., 0.]]))
    torch.testing.assert_close(actual, torch.tensor([[1., .2, 1.]]))
    actual = initial_pose.segment_distance(points, torch.zeros(1, 2), torch.zeros(1, 2))
    torch.testing.assert_close(actual, points.norm(dim=-1))


def test_continuous_front_domain_covers_old_regions_and_full_width(initial_pose):
    start = torch.tensor([[0., 0.], [.01, -.01]])
    goal = torch.tensor([[0., .12], [-.1, -.02]])
    base = torch.tensor([.7, .7])
    radius = torch.tensor([.1, .1])
    points, valid = initial_pose.front_candidates(start, goal, base, radius, torch.Generator().manual_seed(7), 10000)
    assert (points[..., 0] >= torch.minimum(start[:, 0], goal[:, 0])[:, None]).all()
    assert (points[..., 0] < .7).all() and (points[..., 1].abs() < .5).all()
    selected = points[valid]
    assert selected[:, 0].max() > .69 and selected[:, 1].min() < -.49 and selected[:, 1].max() > .49
    # Samples now include both sides, the gap between old bands, and far front.
    for xlow, xhigh in ((.15,.20),(.21,.27),(.28,.30),(.40,.65)):
        for ylow, yhigh in ((-.45,-.2),(.2,.45)):
            assert ((selected[:,0]>xlow)&(selected[:,0]<xhigh)&(selected[:,1]>ylow)&(selected[:,1]<yhigh)).any()
    again=initial_pose.front_candidates(start, goal, base, radius, torch.Generator().manual_seed(7), 10000)
    assert torch.equal(points,again[0]) and torch.equal(valid,again[1])
    distance=initial_pose.segment_distance(points,start,goal)
    assert (distance[valid]>=.1-1e-7).all()


def test_shelf_v1_window_margin_coverage_and_reproducibility(initial_pose):
    start = torch.tensor([[0., 0.], [.01, -.01]])
    goal = torch.tensor([[0., .12], [-.1, -.02]])
    radius = torch.tensor([.10, .10])
    points, valid = initial_pose.shelf_candidates(start, goal, radius, torch.Generator().manual_seed(17), count=10000)
    assert (points[...,0]>=-.199).all() and (points[...,0]<=.299).all()
    assert (points[...,1]>=-.249).all() and (points[...,1]<=.249).all()
    for lo,hi in ((-.19,-.10),(.10,.19),(.21,.27),(.28,.299)):
        for left,right in ((-.249,-.18),(.18,.249)):
            selected=points[valid]
            assert ((selected[:,0]>lo)&(selected[:,0]<hi)&(selected[:,1]>left)&(selected[:,1]<right)).any()
    repeat=initial_pose.shelf_candidates(start, goal, radius, torch.Generator().manual_seed(17), count=10000)
    assert torch.equal(points,repeat[0]) and torch.equal(valid,repeat[1])
    assert (initial_pose.segment_distance(points,start,goal)[valid]>=.10-1e-7).all()


def test_red_diagnostic_is_geometry_not_candidate_category(initial_pose):
    start=torch.tensor([[0.,0.]]).expand(4,-1)
    goal=torch.tensor([[0.,.1]]).expand(4,-1)
    points=torch.tensor([[.18,.05],[.4,.05],[.18,-.2],[.1,.15]])
    assert initial_pose.inside_red_polygon(points,start,goal).tolist()==[True,False,False,False]
    # Degenerate collinear region still has bounded X extent.
    goal=torch.tensor([[.1,0.]]).expand(4,-1)
    points=torch.tensor([[.18,0.],[.4,0.],[.18,.1],[-.1,0.]])
    assert initial_pose.inside_red_polygon(points,start,goal).tolist()==[True,False,False,False]


def test_whole_hand_lane_clear_checks_path_and_both_ends(initial_pose):
    m = model(initial_pose)
    m.hand_boxes = torch.tensor([True])
    center = torch.tensor([[[0., .05, 1.2]], [[0., .12, 1.2]], [[.3, .05, 1.2]]])
    low, high = center-.01, center+.01
    start, goal = torch.zeros(3, 2), torch.tensor([[0., .1]]).expand(3, -1)
    assert m.lane_clear(low, high, start, goal, torch.full((3,), .04)).tolist() == [False,False,True]


def model(initial_pose):
    m = initial_pose.RobotCollisionModel.__new__(initial_pose.RobotCollisionModel)
    m.frames = []
    m.links = torch.tensor([0])
    m.center = torch.zeros(1, 3)
    m.half = torch.tensor([[.1, .02, .03]])
    m.shelf_low = torch.tensor([[-10., -10., -1.]])
    m.shelf_high = torch.tensor([[10., 10., -.5]])
    return m


def test_rotated_link_bounds_and_collision_margin_active_mask(initial_pose):
    m = model(initial_pose)
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
    pos, joints = torch.zeros(1, 3), torch.zeros(1, 6)
    low, high = m.boxes(joints, pos, q)
    torch.testing.assert_close(high, torch.tensor([[[.02, .1, .03]]]), atol=1e-7, rtol=0.)
    torch.testing.assert_close(low, -high)
    obj_low, obj_high = torch.tensor([[[.021, -.01, -.01]]]), torch.tensor([[[.04, .01, .01]]])
    assert not m.clear(joints, pos, q, obj_low, obj_high, torch.ones(1, 1, dtype=torch.bool)).item()
    assert m.clear(joints, pos, q, obj_low, obj_high, torch.zeros(1, 1, dtype=torch.bool)).item()
    pos[:, 2] = -.55
    assert not m.clear(joints, pos, q, obj_low, obj_high, torch.zeros(1, 1, dtype=torch.bool)).item()


def test_articulated_child_uses_joint_frame_and_reset_finger_angle(initial_pose):
    m = model(initial_pose)
    identity, origin = torch.tensor([1., 0., 0., 0.]), torch.zeros(3)
    m.frames = [(0, torch.tensor([.2, 0., 0.]), identity, origin, identity, 2, 0)]
    m.links = torch.tensor([1])
    m.center = torch.tensor([[.1, 0., 0.]])
    joint = torch.tensor([[math.pi/2]])
    low, high = m.boxes(joint, torch.zeros(1, 3), identity[None])
    torch.testing.assert_close((low+high)/2, torch.tensor([[[.2, .1, 0.]]]), atol=1e-7, rtol=0.)


@pytest.mark.parametrize('block_all,skip_grid_batches', [(False,0), (False,1), (False,2), (True,0)])
def test_shelf_v1_recovers_real_budget_exhaustion(initial_pose, monkeypatch, tmp_path,
                                                 block_all, skip_grid_batches):
    """Captured failing layout: real USD FK/collision boxes and Warp arithmetic.

    Freeze the four original XY proposals so the rare failure is deterministic.
    The fixture is one row from seed-42 layout stress, not a policy trajectory.
    """
    import warp as wp
    wp.config.kernel_cache_dir = str(tmp_path / 'warp')
    def decode(x):
        if isinstance(x, dict):
            if 'tensor' in x:
                return torch.tensor(x['tensor'], dtype=getattr(torch, x['dtype']))
            return {k: decode(v) for k, v in x.items()}
        return [decode(v) for v in x] if isinstance(x, list) else x
    data = decode(json.loads((Path(__file__).parent/'fixtures/prof_revision_ik_v1_budget.json').read_text()))
    ik_module = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_ik')
    warp_module = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_warp_ik')
    chain = SimpleNamespace(offset_pos=torch.tensor(data['offset_pos']),
                            offset_quat=torch.tensor(data['offset_quat']))
    chain.joints = [(*frames, int(axis), int(column))
                    for frames, axis, column in zip(zip(*data['ik']['frames']),
                        data['ik']['axes'], data['ik']['columns'])]
    collision = initial_pose.RobotCollisionModel.__new__(initial_pose.RobotCollisionModel)
    collision.__dict__.update(data['model'])
    env = SimpleNamespace(device='cpu', num_envs=1, _initial_pose_collision=collision,
        _initial_pose_rng=torch.Generator().manual_seed(42), _revision_bounds_t=data['bounds'],
        _fast_ik=warp_module.WarpResetArmIK(chain), _fast_initial=data['initial'],
        _fast_limits=data['limits'], _goal_reset_arm_ids=data['arm_ids'],
        scene={'robot':SimpleNamespace(data=SimpleNamespace(default_joint_pos=data['default_joint_pos']))})
    monkeypatch.setattr(initial_pose, 'shelf_candidates', lambda *a, **k: (data['xy'], data['allowed']))
    params = dict(initial_pose_mode='shelf_area_v1', initial_x_bounds=(-.2,.3),
                  front_y_bounds=(-.25,.25), front_clearance=(.08,.10),
                  lateral_offset=(.10,.18), height_jitter=.02)
    if skip_grid_batches:
        # Exercise shrinking/retrying the pending rows beyond the first chunk.
        clear=collision.clear_boxes
        calls=0
        def delayed_clear(*args):
            nonlocal calls
            calls+=1
            result=clear(*args)
            return torch.zeros_like(result) if 3 <= calls < 3+skip_grid_batches else result
        monkeypatch.setattr(collision, 'clear_boxes', delayed_clear)
    if block_all:
        # Even recovery must reject an unsafe scene and terminate in finite work.
        calls=[]
        default_before=env.scene['robot'].data.default_joint_pos.clone()
        solve=env._fast_ik.solve
        def counted(*args):
            calls.append(len(args[0]))
            return solve(*args)
        monkeypatch.setattr(env._fast_ik, 'solve', counted)
        monkeypatch.setattr(collision, 'clear_boxes',
                            lambda low, *args: torch.zeros(len(low), dtype=torch.bool))
        with pytest.raises(RuntimeError, match='Collision-free initial pose budget exhausted'):
            initial_pose.select_initial_pose(env, *data['args'], params)
        assert len(calls) <= 5 and sum(calls) <= 257
        assert torch.equal(env.scene['robot'].data.default_joint_pos, default_before)
        return
    selected, angles, pe, re = initial_pose.select_initial_pose(env, *data['args'], params)
    assert (selected[:,:2] >= torch.tensor([-.199,-.249])).all()
    assert (selected[:,:2] <= torch.tensor([.299,.249])).all()
    assert pe.max() < .0005 and re.max() < .003
    assert env._initial_pose_metrics['Reset/pose_grid_recovery_fraction'] == 1.
    assert env._initial_pose_front_selected.all()
    # Independent call-site verification of every returned link against the scene.
    _, local, yaw, active, sweep, _, _, bp, bq, sp, sq, _ = data['args']
    q = torch.zeros((*yaw.shape,4)); q[...,0]=(yaw/2).cos(); q[...,3]=(yaw/2).sin()
    half = (data['bounds'][...,1,:]-data['bounds'][...,0,:])/2
    centers = local + initial_pose.quat_apply(q, data['bounds'].mean(-2))
    extents = half.clone()
    extents[...,0] = yaw.cos().abs()*half[...,0]+yaw.sin().abs()*half[...,1]
    extents[...,1] = yaw.sin().abs()*half[...,0]+yaw.cos().abs()*half[...,1]
    joints=data['default_joint_pos'].clone();joints[:,data['arm_ids']]=angles
    low, high=collision.boxes(joints, initial_pose.quat_apply_inverse(sq,bp-sp),
                             initial_pose.quat_mul(initial_pose.quat_conjugate(sq),bq))
    assert collision.clear_boxes(low,high,centers-extents,centers+extents,active).all()
    assert collision.lane_clear(low,high,centers[:,0,:2],centers[:,0,:2]+sweep[:,:2],
                                half[:,0,:2].norm(dim=-1)).all()
