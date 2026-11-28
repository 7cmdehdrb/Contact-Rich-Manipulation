"""Wide reset geometry, real hand FK, and selected posture state propagation."""
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def layout(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_layout')


@pytest.fixture
def real_hand(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.common.prof_revision_initial_pose')
    def decode(x):
        if isinstance(x, dict):
            if 'tensor' in x:
                return torch.tensor(x['tensor'], dtype=getattr(torch, x['dtype']))
            return {k: decode(v) for k, v in x.items()}
        return [decode(v) for v in x] if isinstance(x, list) else x
    d = decode(json.loads((Path(__file__).parent/'fixtures/prof_revision_ik_v1_budget.json').read_text()))
    model = m.RobotCollisionModel.__new__(m.RobotCollisionModel)
    model.__dict__.update(d['model'])
    names = [p.rsplit('/', 1)[-1] for p in model.paths]
    def find_joints(requested, preserve_order=True):
        ids = [model.frames[names.index(name.removesuffix('_joint'))-1][-1] for name in requested]
        return ids, requested
    robot = SimpleNamespace(find_joints=find_joints, data=SimpleNamespace(default_joint_pos=d['default_joint_pos']))
    env = SimpleNamespace(device='cpu', num_envs=1, scene={'robot': robot},
        _fast_default_eef=(None, d['args'][-1]), _initial_pose_collision=model,
        _initial_pose_rng=torch.Generator().manual_seed(42), _revision_bounds_t=d['bounds'],
        _fast_initial=d['initial'], _fast_limits=d['limits'], _goal_reset_arm_ids=d['arm_ids'])
    return m, env, model, d


def params():
    return dict(initial_pose_mode='shelf_area_v1', initial_x_bounds=(-.2, .3), front_y_bounds=(-.45, .45),
                front_clearance=(.08, .1), lateral_offset=(.1, .18), height_jitter=.02,
                orientation_bounds_deg=(10., 5., 15.), max_tilt_deg=10., hand_self_clearance_m=.002,
                hand_drive_bounds=((0., .3), (0., .55), (0., .55), (0., .55), (0., .5), (0., .15)))


def test_wide_100k_samples_and_independent_footprint_oracle(layout):
    n = 100000
    rng = torch.Generator().manual_seed(20261128)
    radius = .036+.008*torch.rand((n,), generator=rng)
    bounds = torch.tensor([[[-.04, -.04, 0.], [.04, .04, .1]]]).repeat(n, 17, 1, 1)
    bounds[:, 0, 0, :2], bounds[:, 0, 1, :2] = -radius[:, None], radius[:, None]
    # Construct command-only instance without a large cell grid for 100k rows.
    m = layout.ProfRevisionWideLayout.__new__(layout.ProfRevisionWideLayout)
    m.bounds, m.workspace_inset = bounds, .02
    m.spawn_extra_inset = .06
    p, g = m.sample_commands(torch.arange(n), rng)
    distance = (g-p).norm(dim=-1)
    assert distance.min() >= .06-1e-6 and distance.max() > .7
    for point in (p, g, (p+g)/2):
        assert (point.abs()+radius[:, None] <= torch.tensor([.16, .46])+1e-6).all()
    # Start XY, normalized by the actual size-dependent domain, is area uniform.
    assert (p.abs()+radius[:, None] <= torch.tensor([.10, .40])+1e-6).all()
    normalized = p/(torch.tensor([.10, .40])-radius[:, None])
    assert normalized.mean(0).abs().max() < .01
    assert ((normalized.square().mean(0)-1/3).abs() < .01).all()
    assert ((p[:, 1].abs() > .3).float().mean() > .12)
    # Goals still reach the old wider domain, beyond the reduced start area.
    assert (g[:, 0].abs()+radius > .10).any()
    assert (g[:, 1].abs()+radius > .40).any()


def test_boundary_angles_corners_offsets_and_infeasible_domain(layout):
    p = torch.tensor([[-.12, -.42], [.12, .42], [0., 0.], [-.12, .42], [.12, -.42]])
    lower, upper = torch.tensor([[-.12, -.42]]).expand(5, -1), torch.tensor([[.12, .42]]).expand(5, -1)
    for rank in (0., .25, .5, .999999):
        g = layout.sample_wide_goals(p, lower, upper, .06, torch.full((5,), rank), torch.ones(5))
        assert ((g >= lower-1e-6) & (g <= upper+1e-6)).all()
        assert ((g-p).norm(dim=-1) >= .06-1e-6).all()
    m = layout.ProfRevisionWideLayout.__new__(layout.ProfRevisionWideLayout)
    m.bounds = torch.tensor([[[[-.02, -.03, 0.], [.04, .05, .1]]]]).repeat(1, 17, 1, 1)
    m.workspace_inset = .02
    m.spawn_extra_inset = .06
    lo, hi = m.workspace_bounds(torch.tensor([0]))
    torch.testing.assert_close(lo, torch.tensor([[-.13, -.42]]))
    torch.testing.assert_close(hi, torch.tensor([[.11, .44]]))
    slo, shi = m.spawn_bounds(torch.tensor([0]))
    torch.testing.assert_close(slo, torch.tensor([[-.07, -.36]]))
    torch.testing.assert_close(shi, torch.tensor([[.05, .38]]))
    m.spawn_extra_inset = .13
    with pytest.raises(ValueError, match='no target start area'):
        m.sample_commands(torch.tensor([0]), torch.Generator().manual_seed(42))
    with pytest.raises(ValueError, match='no feasible'):
        layout.sample_wide_goals(torch.zeros(1, 2), torch.full((1, 2), -.01),
                                torch.full((1, 2), .01), .06, torch.zeros(1), torch.zeros(1))


def test_long_goals_keep_full_lane_and_initial_approach_without_convex_hull(layout, monkeypatch, tmp_path):
    import warp as wp
    wp.config.kernel_cache_dir = str(tmp_path/'warp')
    bounds = torch.tensor([[[[-.04, -.04, 0.], [.04, .04, .1]]]]).repeat(3, 17, 1, 1)
    m = layout.ProfRevisionWideLayout(bounds, torch.ones(3, 17), cell_size=.005)
    start = torch.tensor([[-.046, .401], [-.12, -.42], [-.12, .42]])
    goal = torch.tensor([[-.001, -.219], [-.12, .42], [.12, -.42]])
    monkeypatch.setattr(m, 'sample_commands', lambda *a, **k: (start, goal))
    randint = torch.randint
    def count_six(low, high, size, **kwargs):
        return randint(low, high, size, **kwargs).fill_(6)
    monkeypatch.setattr(torch, 'randint', count_six)
    local, yaw, active, delta, requested = m.sample(torch.arange(3), torch.Generator().manual_seed(42))
    assert (requested == 6).all() and (active[:, 1:].sum(-1) == 6).all()
    assert torch.equal(local[:, 0, :2], start) and torch.equal(delta, goal-start)
    # Independent scalar closest-point oracle for every active circle and pair.
    def distance(point, a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = min(1., max(0., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/max(dx*dx+dy*dy, 1e-20)))
        return math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)
    radius = math.sqrt(2)*.04
    for row in range(3):
        centers = local[row, active[row], :2].tolist()
        for i, point in enumerate(centers[1:], 1):
            assert .2-abs(point[0]) >= radius+.01-1e-6
            assert .5-abs(point[1]) >= radius+.01-1e-6
            assert distance(point, start[row].tolist(), goal[row].tolist()) >= .04+radius+.02-1e-6
            assert distance(point, start[row].tolist(), [.2, float(start[row, 1])]) >= .04+radius+.02-1e-6
            for previous in centers[1:i]:
                assert math.dist(point, previous) >= 2*radius+.02-1e-6


def test_obb_sat_cross_axes_and_gap_margin(real_hand):
    m, *_ = real_hand
    q = torch.tensor([[1., 0, 0, 0.]])
    half = torch.tensor([[.1, .02, .03]])
    a, b = torch.zeros(1, 3), torch.tensor([[.2025, 0., 0.]])
    assert m.separated_obbs(a, q, half, b, q, half, .002).all()
    assert not m.separated_obbs(a, q, half, b, q, half, .003).any()
    b[:] = torch.tensor([.08, .08, 0.])
    qb = torch.tensor([[math.cos(math.pi/8), 0., 0., math.sin(math.pi/8)]])
    # Both long rotated boxes cross the same point.
    assert not m.separated_obbs(a, qb, half, b, qb, half).any()


def test_real_hand_mimic_wide_flexion_and_conservative_self_clear(real_hand):
    m, env, model, d = real_hand
    s = m.InitialHandRandomizer(env, model)
    assert s.self_clear(d['default_joint_pos']).all()
    n = 1024
    identity = torch.tensor([1., 0, 0, 0.]).expand(n, -1)
    result = s.sample(d['args'][-1].expand(n, -1), identity, identity, torch.Generator().manual_seed(7), params())
    q, hand, u, rpy, tilt, clear = result
    assert clear.any() and tilt.any()
    assert (u[:, 1:5] > .5).any() and u[:, 5].max() <= .15
    assert ((q.norm(dim=-1)-1).abs() < 1e-5).all()
    assert torch.allclose(hand[:, 5], hand[:, 4]*1.0843)
    # A deliberately excessive curl/opposition must exercise pair rejection.
    closed = d['default_joint_pos'].clone()
    closed[:, s.hand_ids] = (s.upper[s.mimic]*s.ratios)[None]
    assert not s.self_clear(closed).any()
    # Independently compare tip FK between open and a valid strongly curled hand.
    index = ((u[:, 1:5].max(-1).values > .5) & clear & tilt).nonzero()[0, 0]
    joints = d['default_joint_pos'].clone()
    joints[:, s.hand_ids] = hand[index]
    p0, _ = model.link_poses(d['default_joint_pos'], torch.zeros(1, 3), identity[:1])
    p1, _ = model.link_poses(joints, torch.zeros(1, 3), identity[:1])
    names = [p.rsplit('/', 1)[-1] for p in model.paths]
    tips = [names.index(f'inspire_{finger}_force_sensor_3') for finger in ('index', 'middle', 'ring', 'little')]
    assert (p1[:, tips]-p0[:, tips]).norm(dim=-1).max() > .08


def test_orientation_frame_tilt_and_zero_ranges(real_hand):
    m, env, model, d = real_hand
    s = m.InitialHandRandomizer(env, model)
    identity = torch.tensor([[1., 0, 0, 0.]])
    settings = params()
    settings['orientation_bounds_deg'] = (0., 0., 0.)
    q, *_ = s.sample(d['args'][-1], identity, identity, torch.Generator().manual_seed(8), settings)
    assert torch.allclose(q, d['args'][-1], atol=1e-6)
    # Roll10 + pitch5 exceeds the combined 10 degree tilt limit.
    assert math.degrees(math.acos(math.cos(math.radians(10))*math.cos(math.radians(5)))) > 10


def test_random_posture_repeats_and_quaternion_matches_declared_frame(real_hand):
    m, env, model, d = real_hand
    s = m.InitialHandRandomizer(env, model)
    n = 64
    base = torch.tensor([math.cos(.2), 0., 0., math.sin(.2)]).expand(n, -1)
    shelf = torch.tensor([math.cos(.3), 0., 0., math.sin(.3)]).expand(n, -1)
    args = (d['args'][-1].expand(n, -1), base, shelf)
    first = s.sample(*args, torch.Generator().manual_seed(9), params())
    second = s.sample(*args, torch.Generator().manual_seed(9), params())
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    q, _, _, rpy, tilt, _ = first
    # An independent matrix rotation construction checks local R/P and shelf yaw.
    def rx(a):
        c, z = math.cos(a), math.sin(a)
        return torch.tensor([[1., 0., 0.], [0., c, -z], [0., z, c]])
    def ry(a):
        c, z = math.cos(a), math.sin(a)
        return torch.tensor([[c, 0., z], [0., 1., 0.], [-z, 0., c]])
    def rz(a):
        c, z = math.cos(a), math.sin(a)
        return torch.tensor([[c, -z, 0.], [z, c, 0.], [0., 0., 1.]])
    c = m.matrix_from_quat(s.hand_in_eef)
    default = m.matrix_from_quat(d['args'][-1][0])
    for i in range(n):
        roll, pitch, yaw = rpy[i].tolist()
        expected = rz(-.4)@rz(yaw)@rz(.4)@default@c@ry(pitch)@rx(roll)@c.T
        torch.testing.assert_close(m.matrix_from_quat(q[i]), expected, atol=1e-6, rtol=1e-5)
        assert bool(tilt[i]) == (math.cos(roll)*math.cos(pitch) >= math.cos(math.radians(10)))


@pytest.mark.parametrize('block_all', [False, True])
def test_real_warp_selector_carries_posture_through_grid_recovery(real_hand, monkeypatch, tmp_path, block_all):
    import warp as wp
    wp.config.kernel_cache_dir = str(tmp_path/'warp')
    m, env, model, d = real_hand
    warp = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_warp_ik')
    chain = SimpleNamespace(offset_pos=torch.tensor(d['offset_pos']), offset_quat=torch.tensor(d['offset_quat']))
    chain.joints = [(*f, int(axis), int(column)) for f, axis, column in zip(zip(*d['ik']['frames']), d['ik']['axes'], d['ik']['columns'])]
    env._fast_ik = warp.WarpResetArmIK(chain)
    if block_all:
        env.cfg = SimpleNamespace(log_dir=str(tmp_path))
        monkeypatch.setattr(model, 'clear_boxes', lambda low, *args: torch.zeros(len(low), dtype=torch.bool))
        solve = env._fast_ik.solve
        attempts = []
        def counted(*a):
            attempts.append(len(a[0]))
            return solve(*a)
        monkeypatch.setattr(env._fast_ik, 'solve', counted)
        with pytest.raises(RuntimeError, match='budget exhausted'):
            m.select_initial_pose(env, *d['args'], params())
        assert sum(attempts) == 257 and len(attempts) == 5
        dump = torch.load(next(tmp_path.glob('initial_pose_failure_*.pt')), weights_only=False)
        assert dump['failed_env_ids'].tolist() == [0]
        assert 'rng_before' in dump and torch.equal(dump['local'], d['args'][1])
        return
    clear = model.clear_boxes
    calls = 0
    def delay(*a):
        nonlocal calls
        calls += 1
        answer = clear(*a)
        return torch.zeros_like(answer) if calls <= 2 else answer
    monkeypatch.setattr(model, 'clear_boxes', delay)
    selected, angles, pe, re = m.select_initial_pose(env, *d['args'], params())
    assert pe.max() < .0005 and re.max() < .003
    assert env._initial_pose_metrics['Reset/pose_grid_recovery_fraction'] == 1.
    # Reconstruct full hand FK for the exact selected finger values.
    joints = d['default_joint_pos'].clone()
    joints[:, d['arm_ids']] = angles
    joints[:, env._initial_hand_randomizer.hand_ids] = env._initial_pose_hand_targets
    assert env._initial_hand_randomizer.self_clear(joints).all()
    assert (env._initial_pose_hand_drive > .3).any()
    assert torch.isfinite(env._initial_pose_selected_quat_base).all()
    ik = importlib.import_module('_baseline_cpu_pkg.mdp.common.wrench_refactor_ik')
    # The captured Warp fixture encodes fixed columns as -1; Torch FK uses None.
    torch_chain = SimpleNamespace(offset_pos=chain.offset_pos, offset_quat=chain.offset_quat,
        joints=[(*j[:4], j[4] if j[5] >= 0 else None, j[5] if j[5] >= 0 else None) for j in chain.joints])
    actual_pos, actual_quat = ik.ResetArmIK.forward(torch_chain, angles)
    relative = m.quat_mul(m.quat_conjugate(env._initial_pose_selected_quat_base), actual_quat)
    assert (2*torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs()) < .003).all()
    desired_world = d['args'][9]+m.quat_apply(d['args'][10], selected)
    actual_world = d['args'][7]+m.quat_apply(d['args'][8], actual_pos)
    assert (actual_world-desired_world).norm(dim=-1).max() < .0005
