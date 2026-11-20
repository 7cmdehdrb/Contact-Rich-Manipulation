"""Raw-command costs and execution bounds must remain separate."""
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize('group', ['arm', 'hand'])
def test_excess_boundaries_and_normalized_width(load_mdp, group):
    module = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    limits = importlib.import_module('_baseline_cpu_pkg.assets.hand_kinematics')
    lo = torch.full((6,), -math.pi / 2) if group == 'arm' else torch.zeros(6)
    hi = torch.full((6,), math.pi / 2) if group == 'arm' else torch.tensor(limits.HAND_DRIVE_UPPER_LIMITS)
    width = hi - lo
    values = torch.stack([lo, hi, (lo + hi) / 2, lo - width, hi + width, hi + 2 * width])
    raw = torch.zeros(6, 12)
    raw[:, slice(0, 6) if group == 'arm' else slice(6, 12)] = values
    before = raw.clone()
    result = module.action_limit_excess(NS(action_manager=NS(action=raw)), group)
    torch.testing.assert_close(result, torch.tensor([0., 0., 0., 1., 1., 4.]))
    torch.testing.assert_close(raw, before)


def test_raw_rate_sees_different_saturated_commands(load_mdp):
    rewards = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_simple')
    obs = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.stabilize_sweep')
    current = torch.full((2, 12), -200.)
    previous = torch.full((2, 12), -2.)
    env = NS(action_manager=NS(action=current, prev_action=previous))
    for group in ('arm', 'hand'):
        torch.testing.assert_close(rewards.finger_action_rate_l2(env, group), torch.full((2,), 6 * 198. ** 2))
    torch.testing.assert_close(obs.legacy_clipped_previous_action(env), current.clamp(-math.pi / 2, math.pi / 2))
    torch.testing.assert_close(current, torch.full((2, 12), -200.))


def test_arm_clips_before_parent_without_mutating_input(load_mdp):
    # Execute the actual child method against a recording OSC parent; physics is
    # exercised separately by scripts/check_sweep_only_action_penalty.py.
    from conftest import MODULE_ROOT
    from test_omni_configuration import _execute_definitions

    class Parent:
        def process_actions(self, actions):
            self.received = actions.clone()

    scope = _execute_definitions(
        MODULE_ROOT / 'mdp/action_functions/wrench_refactor.py',
        {'WrenchRefactorOSCAction': Parent, 'math': math}, {'SweepOnlyClippedOSCAction'})
    term = scope.SweepOnlyClippedOSCAction()
    raw = torch.tensor([[-200., -math.pi / 2, -.2, .2, math.pi / 2, 200.]])
    before = raw.clone()
    term.process_actions(raw)
    torch.testing.assert_close(term.received, raw.clamp(-math.pi / 2, math.pi / 2))
    torch.testing.assert_close(raw, before)


def test_hand_limits_and_mimic_preserve_raw_input(load_mdp):
    action = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.sweep_simple')
    kin = importlib.import_module('_baseline_cpu_pkg.assets.hand_kinematics')
    term = object.__new__(action.FingerJointPositionAction)
    term._upper_limits = torch.tensor(kin.HAND_DRIVE_UPPER_LIMITS)
    term._drive_indices = torch.tensor(kin.HAND_MIMIC_DRIVE_INDICES)
    term._multipliers = torch.tensor(kin.HAND_MIMIC_MULTIPLIERS)
    term._raw_actions = torch.zeros(2, 6)
    term._processed_actions = torch.zeros(2, 6)
    term._joint_targets = torch.zeros(2, 12)
    raw = torch.stack([torch.full((6,), -200.), torch.full((6,), 200.)])
    term.process_actions(raw)
    torch.testing.assert_close(term.raw_actions, raw)
    torch.testing.assert_close(term.processed_actions[0], torch.zeros(6))
    torch.testing.assert_close(term.processed_actions[1], term._upper_limits)
    torch.testing.assert_close(term._joint_targets, term.processed_actions[:, term._drive_indices] * term._multipliers)


def test_linear_reach_live_rotated_center_boundaries_and_no_reset_scaling(load_mdp):
    module = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    # Rotate a nonzero bounds-center offset: measuring the link origin or an
    # episode-fixed surface point would produce different rewards here.
    distances = torch.tensor([0., .05, .10, .100001, .15, .20, .30, .35, .449999, .45, .60],
                             dtype=torch.float64)
    n = len(distances)
    root = torch.tensor([.2, -.1, 1.05], dtype=torch.float64).expand(n, 1, 3).clone()
    rotation = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)], dtype=torch.float64).expand(n, 1, 4)
    bounds = torch.tensor([[.01, -.01, 0.], [.03, .01, .10]], dtype=torch.float64).expand(n, 1, 2, 3)
    center = root[:, 0] + torch.tensor([0., .02, .05], dtype=torch.float64)
    eef = center.clone(); eef[:, 0] += distances
    objects = NS(data=NS(object_link_pos_w=root, object_link_quat_w=rotation))
    env = NS(scene={'object_collection': objects, 'ee_frame': NS(data=NS(target_pos_w=eef[:, None]))},
             _revision_bounds_t=bounds)
    expected = torch.tensor([1., 1., 1., 1.-.000001/.35, 6./7., 5./7., 3./7., 2./7.,
                             .000001/.35, 0., 0.], dtype=torch.float64)
    actual = module.linear_reach(env)
    torch.testing.assert_close(actual, expected)
    # The stateless reward needs neither a reset snapshot nor episode duration.
    env.cfg = NS(episode_length_s=500.)
    env._initial_distance = torch.full((n,), 100.)
    torch.testing.assert_close(module.linear_reach(env), actual)
    root[:, :, 0] += .1
    assert not torch.allclose(module.linear_reach(env), actual)


@pytest.mark.parametrize('params', [
    {'near_radius': -.1}, {'near_radius': float('nan')}, {'near_radius': float('inf')},
    {'decay_distance': 0.}, {'decay_distance': -.1},
    {'decay_distance': float('nan')}, {'decay_distance': float('inf')},
])
def test_linear_reach_rejects_invalid_distance_parameters(load_mdp, params):
    module = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.stabilize_sweep')
    with pytest.raises(ValueError):
        module.linear_reach(NS(), **params)
