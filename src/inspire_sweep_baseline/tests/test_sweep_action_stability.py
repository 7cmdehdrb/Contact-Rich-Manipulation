"""Normalized hand commands and finite raw-command costs in the Sweep family."""
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def modules(load_mdp):
    return NS(
        action=importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.sweep_simple'),
        kin=importlib.import_module('_baseline_cpu_pkg.assets.hand_kinematics'),
        rate=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_simple'),
        limit=importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep'),
    )


def hand_term(modules, cls):
    term = object.__new__(cls)
    term._upper_limits = torch.tensor(modules.kin.HAND_DRIVE_UPPER_LIMITS)
    term._drive_indices = torch.tensor(modules.kin.HAND_MIMIC_DRIVE_INDICES)
    term._multipliers = torch.tensor(modules.kin.HAND_MIMIC_MULTIPLIERS)
    term._raw_actions = torch.zeros(5, 6)
    term._processed_actions = torch.zeros(5, 6)
    term._joint_targets = torch.zeros(5, 12)
    return term


def test_normalized_hand_clamps_maps_each_drive_and_preserves_raw(modules):
    term = hand_term(modules, modules.action.NormalizedFingerJointPositionAction)
    raw = torch.tensor([[-100.] * 6, [0.] * 6, [1.] * 6, [100.] * 6,
                        [0., .2, .4, .6, .8, 1.]])
    before = raw.clone()
    term.process_actions(raw)
    expected = raw.clamp(0, 1) * term._upper_limits
    torch.testing.assert_close(term.processed_actions, expected)
    torch.testing.assert_close(term.raw_actions, before)
    torch.testing.assert_close(raw, before)
    torch.testing.assert_close(term._joint_targets,
                               expected[:, term._drive_indices] * term._multipliers)
    # Radian-based siblings keep their original meaning, even for in-range input.
    parent = hand_term(modules, modules.action.FingerJointPositionAction)
    parent.process_actions(raw)
    torch.testing.assert_close(parent.processed_actions,
                               raw.clamp_min(0).minimum(term._upper_limits))
    assert not torch.allclose(parent.processed_actions[-1], expected[-1])


@pytest.mark.parametrize('group', ['arm', 'hand'])
def test_rate_cap_preserves_normal_cost_and_caps_extreme_changes(modules, group):
    current = torch.zeros(3, 12)
    selection = slice(0, 6) if group == 'arm' else slice(6, 12)
    current[1, selection] = 1.
    current[2, selection] = 1e20
    env = NS(action_manager=NS(action=current, prev_action=torch.zeros_like(current)))
    actual = modules.rate.finger_action_rate_l2(env, group, max_cost=100.)
    torch.testing.assert_close(actual, torch.tensor([0., 6., 100.]))
    # No ceiling is introduced into other environments' default calls.
    assert torch.isinf(modules.rate.finger_action_rate_l2(env, group)[2])


@pytest.mark.parametrize('group', ['arm', 'hand'])
def test_limit_normalized_boundaries_and_cap(modules, group):
    lo = torch.full((6,), -math.pi / 2) if group == 'arm' else torch.zeros(6)
    hi = torch.full((6,), math.pi / 2) if group == 'arm' else torch.ones(6)
    width = hi - lo
    values = torch.stack([lo, hi, lo - .1 * width, hi + .5 * width,
                          hi + width, lo - 10 * width, torch.full((6,), 1e20)])
    raw = torch.zeros(7, 12)
    raw[:, slice(0, 6) if group == 'arm' else slice(6, 12)] = values
    before = raw.clone()
    actual = modules.limit.action_limit_excess(NS(action_manager=NS(action=raw)), group,
                                               hand_normalized=True, max_cost=10.)
    torch.testing.assert_close(actual, torch.tensor([0., 0., .01, .25, 1., 10., 10.]))
    torch.testing.assert_close(raw, before)
    # Manager applies strengthened weight and policy dt once.
    torch.testing.assert_close(-3. * .02 * actual,
                               torch.tensor([0., 0., -.0006, -.015, -.06, -.6, -.6]))


@pytest.mark.parametrize('maximum', [0., -1., float('nan'), float('inf')])
def test_invalid_cost_ceiling_is_rejected(modules, maximum):
    env = NS(action_manager=NS(action=torch.zeros(1, 12), prev_action=torch.zeros(1, 12)))
    with pytest.raises(ValueError, match='max_cost'):
        modules.rate.finger_action_rate_l2(env, max_cost=maximum)
    with pytest.raises(ValueError, match='max_cost'):
        modules.limit.action_limit_excess(env, 'hand', hand_normalized=True, max_cost=maximum)
