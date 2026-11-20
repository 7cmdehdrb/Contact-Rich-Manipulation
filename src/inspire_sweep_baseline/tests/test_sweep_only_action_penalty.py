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
