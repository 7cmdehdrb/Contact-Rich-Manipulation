"""Exercise v1 production tensor/state/reward code without starting Isaac Sim."""
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def modules(load_mdp):
    return NS(
        state=importlib.import_module('_baseline_cpu_pkg.mdp.common.sweep_v0_state'),
        side=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_contact_preference'),
        reward=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep'),
        termination=importlib.import_module('_baseline_cpu_pkg.mdp.termination_functions.sweep'),
    )


def initialize(modules, angles=(0., 90., 180., -90.), dtype=torch.float32):
    n = len(angles)
    state = modules.state.SingleContactV1SweepState(n, 'cpu')
    radians = torch.tensor(angles, dtype=dtype) * math.pi / 180.
    direction = torch.stack((-radians.cos(), radians.sin(), radians * 0.), -1)
    center = torch.zeros(n, 3, dtype=dtype)
    hand = center.clone()
    hand[:, 2] = .2
    bounds = torch.tensor([[[-.03, -.04, -.05], [.03, .04, .05]]], dtype=dtype).expand(n, -1, -1)
    quat = torch.tensor([[1., 0., 0., 0.]], dtype=dtype).expand(n, -1)
    ids = torch.arange(n)
    state.reset(ids, center, quat, direction)
    state.reset_reach(ids, center, hand, bounds, direction, modules.side.sweep_region(direction), quat)
    return state, center, hand, bounds, direction, quat


def test_region_three_fixed_point_and_other_regions_opposite(modules):
    state, center, hand, bounds, direction, quat = initialize(modules, (0, 90, 150.01, 180, -150.01, -90))
    assert state.reach_region.tolist() == [1, 2, 3, 3, 3, 4]
    expected = -direction * .05
    expected[2:5] = torch.tensor([-.05 * math.cos(math.pi / 12), .05 * math.sin(math.pi / 12), 0.])
    torch.testing.assert_close(state.reach_offset, expected)
    # All four XY corners lie on the enclosing circle; no double-applied scale.
    torch.testing.assert_close(state.reach_radius, torch.full((6,), .05))
    torch.testing.assert_close(state.reach_reference_angle[2:5], torch.full((3,), math.pi / 12))
    # Reset radius reflects new scaled bounds, only on the selected row.
    state.reset_reach(torch.tensor([0]), center[:1], hand[:1], bounds[:1] * 2,
                      direction[:1], state.reach_region[:1], quat[:1])
    torch.testing.assert_close(state.reach_radius, torch.tensor([.1, .05, .05, .05, .05, .05]))


def test_shelf_frame_rotation_and_live_translation_only(modules, tensor_math):
    state, center, hand, bounds, direction, _ = initialize(modules)
    shelf = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]]).expand(4, -1)
    original = state.reach_offset.clone()
    state.reset_reach(torch.arange(4), center, hand, bounds, direction, state.reach_region, shelf)
    torch.testing.assert_close(state.reach_offset, tensor_math.quat_apply(shelf, original))
    offset = state.reach_offset.clone()
    moved = center + torch.tensor([.1, -.2, .3])
    state.update_reach(moved, hand, 1)
    torch.testing.assert_close(state.reach_reference_point, moved + offset)
    # Neither the live object orientation nor hand position is an input to offset updates.
    state.update_reach(moved, hand + 1., 2)
    torch.testing.assert_close(state.reach_reference_point, moved + offset)


@pytest.mark.parametrize('distance,expected', [(.049999, 0.), (.05, 0.), (.050001, .000001)])
def test_five_cm_boundary_is_three_dimensional(modules, distance, expected):
    state, center, *_ = initialize(modules, (0,))
    hand = state.reach_reference_point.clone()
    hand[:, 2] += distance
    state.update_reach(center, hand, 1)
    assert state.reach_excess.item() == pytest.approx(expected, abs=1e-8)


def test_reach_first_step_signed_round_trip_following_and_duplicate_reads(modules):
    state, center, hand, *_ = initialize(modules, (0,))
    initial = state.reach_excess.clone()
    # Move exactly 1cm toward the full 3D approach point.
    unit = (hand - state.reach_reference_point) / state.reach_distance[:, None]
    closer = hand - .01 * unit
    delta = state.update_reach(center, closer, 1).clone()
    assert (4.9 * delta).item() == pytest.approx(.049, abs=1e-6)
    torch.testing.assert_close(state.update_reach(center, closer, 1), delta)
    back = state.update_reach(center, hand, 2).clone()
    torch.testing.assert_close(delta + back, torch.zeros(1))
    torch.testing.assert_close(state.reach_excess, initial)
    shift = torch.tensor([[.04, -.03, .02]])
    torch.testing.assert_close(state.update_reach(center + shift, hand + shift, 3), torch.zeros(1), atol=1e-7, rtol=0.)
    # The object moves away; following it later earns the symmetric recovery.
    away = state.update_reach(center + shift + torch.tensor([[0., 0., -.02]]), hand + shift, 4).clone()
    assert away.item() < 0.
    recovery = state.update_reach(center + shift + torch.tensor([[0., 0., -.02]]),
                                 hand + shift + torch.tensor([[0., 0., -.02]]), 5)
    torch.testing.assert_close(away + recovery, torch.zeros(1), atol=1e-7, rtol=0.)


def test_partial_reset_preserves_other_row_and_excludes_teleport(modules):
    state, center, hand, bounds, direction, quat = initialize(modules)
    state.update_reach(center, hand - torch.tensor([0., 0., .01]), 10)
    before = {name: value.clone() for name, value in vars(state).items() if isinstance(value, torch.Tensor)}
    ids = torch.tensor([0])
    moved = center[:1] + 10.
    state.reset(ids, moved, quat[:1], direction[:1] + moved)
    state.reset_reach(ids, moved, hand[:1] + 10., bounds[:1], direction[:1], torch.tensor([1]), quat[:1])
    for name, old in before.items():
        torch.testing.assert_close(getattr(state, name)[1:], old[1:])
    center[0] = moved[0]
    hand[0] += 10.
    assert state.update_reach(center, hand, 11)[0].item() == pytest.approx(0., abs=1e-6)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_real_reach_callable_ignores_object_rotation_at_fixed_bounds_center(modules, tensor_math, axis):
    state, center, hand, bounds, direction, quat = initialize(modules, (180,))
    # Nonzero local bounds center distinguishes it from the object's link origin.
    bounds = bounds.clone() + torch.tensor([.01, .02, .03])
    local_center = bounds.mean(1)
    data = NS(object_link_pos_w=(center - local_center)[:, None], object_link_quat_w=quat[:, None].clone())
    env = NS(_stabilize_state=state, _revision_bounds_t=bounds[:, None], step_dt=.02,
             common_step_counter=1, scene={'object_collection': NS(data=data),
                 'ee_frame': NS(data=NS(target_pos_w=hand[:, None]))})
    state.reset_reach(torch.tensor([0]), center, hand, bounds, direction, torch.tensor([3]), quat)
    assert modules.reward.incremental_approach_reach(env).item() == pytest.approx(0., abs=1e-6)
    euler = torch.zeros(3)
    euler[axis] = .7
    rotation = tensor_math.quat_from_euler_xyz(euler[0:1], euler[1:2], euler[2:3])
    data.object_link_quat_w = rotation[:, None]
    data.object_link_pos_w = (center - tensor_math.quat_apply(rotation, local_center))[:, None]
    env.common_step_counter += 1
    assert modules.reward.incremental_approach_reach(env).item() == pytest.approx(0., abs=1e-6)
    torch.testing.assert_close(state.reach_reference_point, center + state.reach_offset)
    # Changing the shelf pose later cannot rotate this episode-fixed offset.
    point = state.reach_reference_point.clone()
    env.scene['shelf'] = NS(data=NS(root_quat_w=rotation))
    env.common_step_counter += 1
    modules.reward.incremental_approach_reach(env)
    torch.testing.assert_close(state.reach_reference_point, point)


@pytest.mark.parametrize('bad', ['bounds', 'direction', 'hand'])
def test_invalid_reach_geometry_is_diagnosed(modules, bad):
    state, center, hand, bounds, direction, quat = initialize(modules, (0,))
    bounds = bounds.clone()
    if bad == 'bounds':
        bounds.zero_()
    elif bad == 'direction':
        direction.zero_()
    else:
        hand.fill_(float('nan'))
    with pytest.raises(ValueError, match='envs \\[0\\]'):
        state.reset_reach(torch.tensor([0]), center, hand, bounds, direction, torch.tensor([1]), quat)


def finish(state, angles, *, force=2., vertical=0., contact=True, valid=True, height=1.1, progress=.01):
    n = len(angles)
    a = torch.tensor(angles) * math.pi / 180.
    f = torch.stack((a.cos() * force, a.sin() * force, torch.full_like(a, vertical)), -1)
    state.previous_distance.fill_(1. + progress)
    state.finish(torch.zeros(n, 3), torch.tensor([[1., 0., 0.]]).expand(n, -1), f,
                 torch.full((n,), contact), torch.full((n,), valid),
                 torch.full((n,), height), torch.ones(n))


def test_direction_curve_monotone_signed_and_vertical_invariant(modules):
    angles = list(range(181))
    state = modules.state.SingleContactV1SweepState(len(angles), 'cpu')
    state.contact_history.fill_(True)
    finish(state, angles)
    a = state.sweep_direction_coefficient.clone()
    assert a[0].item() == 1. and a[-1].item() == -1.
    assert a[30].item() == pytest.approx(0., abs=1e-6)
    assert a[15].item() == pytest.approx(.91520656, abs=1e-6)
    assert a[45].item() == pytest.approx(2. * 2. ** (-(45. / 30.) ** 4) - 1., abs=1e-6)
    assert ((a[1:] - a[:-1]) <= 1e-6).all()
    assert ((a >= -1.) & (a <= 1.)).all()
    finish(state, angles, vertical=100.)
    torch.testing.assert_close(state.sweep_direction_coefficient, a)


@pytest.mark.parametrize('kwargs,usable', [({}, True), ({'force': .999}, False), ({'force': 1.}, True),
    ({'force': 0.}, False), ({'force': float('nan')}, False), ({'vertical': float('nan')}, False),
    ({'contact': False}, False), ({'valid': False}, False), ({'height': 1.12}, False)])
def test_current_force_validity_height_and_history_gates(modules, kwargs, usable):
    state = modules.state.SingleContactV1SweepState(1, 'cpu')
    state.contact_history.fill_(True)
    finish(state, [0.], **kwargs)
    assert state.sweep_gate.item() == usable
    assert state.signed_progress.item() == pytest.approx(.01 if usable else 0., abs=1e-6)
    assert torch.isfinite(state.signed_progress).all()
    state.contact_history.zero_()
    finish(state, [0.])
    assert not state.sweep_gate.item()


@pytest.mark.parametrize('progress', [-.01, 0.])
def test_bad_angle_never_inverts_retreat_or_rewards_stationary_contact(modules, progress):
    state = modules.state.SingleContactV1SweepState(1, 'cpu')
    state.contact_history.fill_(True)
    finish(state, [90.], progress=progress)
    assert state.signed_progress.item() == 0.
    assert state.retreat.item() == pytest.approx(max(-progress, 0.), abs=1e-6)


def test_goal_zero_masks_angle(modules):
    state = modules.state.SingleContactV1SweepState(1, 'cpu')
    state.contact_history.fill_(True)
    state.finish(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3),
                 torch.tensor([True]), torch.tensor([True]), torch.tensor([1.1]), torch.ones(1))
    assert not state.sweep_gate.item() and state.signed_progress.item() == 0.


def test_positive_surface_preference_negative_unit_cost_and_cached_loads(modules):
    state, *_ = initialize(modules)
    state.signed_progress[:] = torch.tensor([.01, -.01, .01, -.01])
    calls = []
    term = object.__new__(modules.side.SignedContactSideSweepProgress)
    term._step, term._value = -1, torch.zeros(4)
    term._surface_loads = lambda env: (calls.append(1) or torch.ones(4), torch.zeros(4))
    env = NS(_stabilize_state=state, step_dt=.02, common_step_counter=1)
    expected = torch.tensor([.35, -.5, .75, -.5])
    torch.testing.assert_close(term(env), expected)
    torch.testing.assert_close(term(env), expected)
    assert len(calls) == 1
    env.common_step_counter += 1
    term(env)
    assert len(calls) == 2


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_joint_cost_reads_actual_termination_and_is_dt_independent(modules, dt):
    velocities = torch.tensor([[.999, 0.], [1., -1.], [1.001, 0.], [2., -2.]])
    robot = NS(data=NS(joint_vel=velocities))
    env = NS(scene={'robot': robot}, step_dt=dt)
    actual = modules.termination.arm_velocity(env, asset_cfg=NS(name='robot', joint_ids=slice(None)), threshold=1.)
    assert actual.tolist() == [False, False, True, True]
    env.termination_manager = NS(get_term=lambda name: actual if name == 'arm_joint_velocity' else None)
    torch.testing.assert_close(-3. * dt * modules.reward.arm_joint_failure(env), torch.tensor([0., 0., -3., -3.]))
    # Other terminations or timeout cannot trigger this penalty.
    actual.zero_()
    assert modules.reward.arm_joint_failure(env).eq(0).all()
