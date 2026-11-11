"""Sensor-origin wrench OSC against the checked-in native controller, on CPU."""
import copy
import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_sweep_simple import family, configs, omni_config
from test_final_configuration import assert_ordered_equal
from test_omni_configuration import _execute_definitions
from test_fusion_holding_osc import osc


def test_wrench_env_parent_isolation_and_registration(family):
    parent = family.SweepSimpleFingerEnvCfg()
    before = parent.to_dict()
    cls = family.SweepSimpleFingerWrenchEnvCfg
    assert cls.__bases__ == (family.SweepSimpleFingerEnvCfg,)
    child = cls()
    expected = copy.deepcopy(before)
    # Action config leaves are exercised with live configclass in test_finger.
    expected['actions']['arm_action']['body_offset'] = None
    expected['terminations']['arm_joint_velocity']['params']['threshold'] = 1.
    expected['curriculum']['arm_joint_velocity_threshold'] = None
    expected['observations']['policy']['actions']['func'] = child.to_dict()['observations']['policy']['actions']['func']
    assert child.observations.policy.actions.func.__name__ == 'wrench_previous_action'
    assert_ordered_equal(child.to_dict(), expected)
    assert child.terminations.arm_joint_velocity.params['threshold'] == 1.
    assert child.curriculum.arm_joint_velocity_threshold is None
    assert child.rewards.arm_joint_velocity.weight == -.03
    child.actions.arm_action.body_name = 'changed'
    assert_ordered_equal(parent.to_dict(), before)
    assert_ordered_equal(family.SweepSimpleFingerEnvCfg().to_dict(), before)
    entries = {}
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym': NS(
        registry=entries, register=lambda **kw: entries.update({kw['id']: kw}))})
    entry = entries['Inspire-Sweep-Simple-Finger-v0-Wrench']
    assert entry['entry_point'] == entries['Inspire-Sweep-Simple-Finger-v0']['entry_point']
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':SweepSimpleFingerWrenchEnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepSimpleFingerWrenchPPORunnerCfg')


def test_native_wrench_rotation_origin_invariance_and_effort_submission(osc, tensor_math):
    shared = _execute_definitions(MODULE_ROOT/'mdp/action_functions/contact_sweep_clutter_fusion_holding_osc.py',
        dict(torch=torch, math_utils=tensor_math, OperationalSpaceControllerAction=osc.action, Sequence=list))
    cls = _execute_definitions(MODULE_ROOT/'mdp/action_functions/sweep_simple_osc.py',
        dict(torch=torch, quat_apply_inverse=tensor_math.quat_apply_inverse,
             UR10OperationalSpaceControllerAction=shared.UR10OperationalSpaceControllerAction),
        {'SensorFrameWrenchOSCAction'}).SensorFrameWrenchOSCAction
    cfg = osc.v0().actions.arm_action
    cfg.effort_limit_scale, cfg.minimum_effort_limit, cfg.saturation_tolerance = 1., 1.e-6, 1.e-6
    cc = cfg.controller_cfg
    cc.target_types = ['wrench_abs']
    cc.motion_control_axes_task = (0,)*6
    cc.contact_wrench_control_axes_task = (1,)*6
    cc.inertial_dynamics_decoupling = False
    cc.gravity_compensation = True
    cc.contact_wrench_stiffness_task = None
    cc.nullspace_control = 'none'
    term = cls.__new__(cls)
    term.cfg = cfg
    term.num_envs = 4
    term._osc = osc.controller(cc, 4, 'cpu')
    term._ee_pose_b = torch.tensor([
        [.3, -.2, .5, 1., 0., 0., 0.], [4., 2., -3., 1., 0., 0., 0.],
        [.3, -.2, .5, math.sqrt(.5), 0., 0., math.sqrt(.5)],
        [4., 2., -3., math.sqrt(.5), 0., 0., math.sqrt(.5)],
    ])
    # Nonzero COM catches missing sensor-to-COM moment conversion.
    term._sensor_com_offset = torch.tensor([[.01, -.02, .03]]).expand(4, -1).clone()
    term._compute_ee_pose = lambda: None
    term._task_frame_pose_b = term._ee_pose_b.clone()
    term._task_frame_transformer = None
    term._raw_actions = torch.zeros(4, 6)
    term._processed_actions = torch.zeros(4, 6)
    term._pose_rel_idx = term._pose_abs_idx = term._stiffness_idx = term._damping_ratio_idx = None
    term._wrench_abs_idx = 0
    term._wrench_scale = torch.tensor([10., 10., 10., 1., 1., 1.])
    term._raw_wrench_limits = torch.tensor([20., 20., 36., 8., 8., 8.])
    raw = torch.tensor([[.2, -.3, .4, .5, -.6, .7]]).expand(4, -1).clone()
    snapshot = raw.clone()
    term.process_actions(raw)
    torch.testing.assert_close(raw, snapshot)
    torch.testing.assert_close(term._raw_actions, raw)
    force_s, moment_s = (raw*term._wrench_scale).split(3, dim=-1)
    q = term._ee_pose_b[:, 3:]
    expected = torch.cat((tensor_math.quat_apply(q, force_s), tensor_math.quat_apply(
        q, moment_s-torch.cross(term._sensor_com_offset, force_s, dim=-1))), -1)
    torch.testing.assert_close(term._osc.desired_ee_wrench_b, expected, atol=2.e-6, rtol=1.e-5)
    torch.testing.assert_close(expected[0], expected[1])
    torch.testing.assert_close(expected[2], expected[3])
    assert not torch.allclose(expected[0], expected[2])
    # Native J^T wrench + gravity, then the original Finger effort limiter.
    term._jacobian_b = torch.eye(6).expand(4, -1, -1).clone()
    term._ee_vel_b = term._ee_force_b = None
    term._mass_matrix = torch.eye(6).expand(4, -1, -1).clone()
    term._gravity = torch.ones(4, 6)*.1
    term._joint_pos = term._joint_vel = torch.zeros(4, 6)
    term._nullspace_joint_pos_target = None
    term._joint_efforts = torch.zeros(4, 6)
    term._torque_saturated = torch.zeros(4, dtype=torch.bool)
    term._joint_ids = list(range(6))
    targets = []
    term._asset = NS(data=NS(joint_effort_limits=torch.full((4, 6), 100.)),
                     set_joint_effort_target=lambda values, joint_ids: targets.append(values.clone()))
    for name in ('_compute_dynamic_quantities', '_compute_ee_jacobian', '_compute_ee_velocity',
                 '_compute_ee_force', '_compute_joint_states'):
        setattr(term, name, lambda: None)
    term.apply_actions()
    torch.testing.assert_close(targets[-1], expected + .1, atol=2.e-6, rtol=1.e-5)
    assert not term.torque_saturated.any()
    term._asset.data.joint_effort_limits[:] = .2
    term.apply_actions()
    assert term.torque_saturated.all()
    assert targets[-1].abs().max() <= .2
    # Zero command must not retain the previous moment/force, only gravity.
    term.process_actions(torch.zeros_like(raw))
    term.apply_actions()
    torch.testing.assert_close(targets[-1], term._gravity)
    large = torch.tensor([[100., -100., 100., -100., 100., -100.]]).expand(4, -1).clone()
    term.process_actions(large)
    accepted = large.clamp(-term._raw_wrench_limits, term._raw_wrench_limits)
    torch.testing.assert_close(term._raw_actions, accepted)
    assert large[0, 0] == 100.  # PPO sample is not modified in-place.
    force_s, moment_s = (accepted*term._wrench_scale).split(3, dim=-1)
    expected_clipped = torch.cat((tensor_math.quat_apply(q, force_s), tensor_math.quat_apply(
        q, moment_s-torch.cross(term._sensor_com_offset, force_s, dim=-1))), -1)
    # Float32 cancellation at a 4 m root offset and 360 N is below 0.3 mNm.
    torch.testing.assert_close(term._osc.desired_ee_wrench_b, expected_clipped, atol=3.e-4, rtol=1.e-5)
    term._osc.desired_ee_wrench_b[0, 0] = float('nan')
    term.apply_actions()
    assert targets[-1][0].count_nonzero() == 0
    assert term.torque_saturated[0]


def test_wrench_last_action_uses_accepted_raw_hand_and_arm(load_mdp):
    hand_module = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.sweep_simple')
    obs = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.sweep_simple')
    k = importlib.import_module('_baseline_cpu_pkg.assets.hand_kinematics')
    hand = hand_module.BoundedFingerJointPositionAction.__new__(hand_module.BoundedFingerJointPositionAction)
    hand._upper_limits = torch.tensor(k.HAND_DRIVE_UPPER_LIMITS)
    hand._drive_indices = torch.tensor(k.HAND_MIMIC_DRIVE_INDICES)
    hand._multipliers = torch.tensor(k.HAND_MIMIC_MULTIPLIERS)
    hand._raw_actions = torch.zeros(2, 6)
    hand._processed_actions = torch.zeros(2, 6)
    hand._joint_targets = torch.zeros(2, 12)
    raw = torch.tensor([[100.]*6, [-100.]*6])
    hand.process_actions(raw)
    expected_hand = torch.stack((hand._upper_limits, torch.zeros(6)))
    torch.testing.assert_close(hand.raw_actions, expected_hand)
    torch.testing.assert_close(hand.processed_actions, expected_hand)
    arm = NS(raw_actions=torch.tensor([[20., -20., 36., -8., 8., -8.], [0.]*6]))
    terms = {'arm_action': arm, 'gripper_action': hand}
    env = NS(action_manager=NS(get_term=terms.__getitem__))
    observed = obs.wrench_previous_action(env)
    torch.testing.assert_close(observed, torch.cat((arm.raw_actions, expected_hand), -1))
    assert observed[0, 0] == 20.  # Neither tanh nor physical scaling.
    assert raw[0, 0] == 100.
    # Reset-style term histories must produce zero Last Action.
    arm.raw_actions.zero_()
    hand._raw_actions.zero_()
    assert obs.wrench_previous_action(env).count_nonzero() == 0
    assert observed[0, 0] == 20.  # Observation does not alias command history.
