"""Finger behavior and parent isolation on CPU; no PhysX simulation."""
import ast
import copy
import importlib
import importlib.util
import math
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_sweep_simple import family, configs, omni_config
from test_final_configuration import assert_ordered_equal
from test_omni_configuration import _execute_definitions


def test_config_parent_preserved(family):
    parent = family.SweepSimpleV1EnvCfg()
    before = parent.to_dict()
    cls = family.SweepSimpleFingerEnvCfg
    assert cls.__bases__ == (family.SweepSimpleV1EnvCfg,)
    child = cls()
    expected = copy.deepcopy(before)
    # Actions use doubles in this fixture, exercised separately below.
    expected['observations']['policy']['hand_configuration'] = child.to_dict()['observations']['policy']['hand_configuration']
    for name in ('arm_action_rate', 'hand_action_rate'):
        expected['rewards'][name]['func'] = child.to_dict()['rewards'][name]['func']
        assert child.rewards.__dict__[name].func.__name__ == 'finger_action_rate_l2'
    assert_ordered_equal(child.to_dict(), expected)
    obs = child.observations.policy
    assert obs.hand_configuration.params['asset_cfg'].joint_names == [
        'inspire_left_thumb_2_joint', 'inspire_left_index_1_joint',
        'inspire_left_middle_1_joint', 'inspire_left_ring_1_joint',
        'inspire_left_little_1_joint', 'inspire_left_thumb_1_joint',
    ]
    assert obs.hand_configuration.params['asset_cfg'].preserve_order
    assert obs.actions.func is parent.observations.policy.actions.func
    child.rewards.hand_action_rate.weight = -999
    child.observations.policy.hand_configuration.params['asset_cfg'].joint_names.reverse()
    assert_ordered_equal(family.SweepSimpleV1EnvCfg().to_dict(), before)


def test_action_mapping_limits_raw_observation_and_partial_reset(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.sweep_simple')
    robot_module = sys.modules['_baseline_cpu_pkg.assets.robot']
    names = list(robot_module.HAND_JOINT_NAMES)
    # Shuffled articulation order prevents accidentally passing by fixed indices.
    articulation_names = names[::-1]
    ids = [articulation_names.index(n) for n in names]
    targets = torch.full((3, 12), -99.)
    def set_target(values, joint_ids, env_ids=None):
        rows = torch.arange(3) if env_ids is None else env_ids
        targets[rows[:, None], torch.tensor(joint_ids)[None, :]] = values
    robot = NS(find_joints=lambda n, preserve_order: (ids, names),
               set_joint_position_target=set_target,
               data=NS(joint_pos=torch.randn(3, 12)))
    env = NS(scene={'robot':robot}, device='cpu', num_envs=3)
    term = m.FingerJointPositionAction(NS(asset_name='robot'),env)
    command = torch.tensor([[.1,.2,.3,.4,.5,.6], [-1.,4.,3.,2.,1.,5.], [.2,.3,.4,.5,.6,.7]])
    term.process_actions(command)
    term.apply_actions()
    assert term.action_dim == 6
    torch.testing.assert_close(term.raw_actions, command)
    torch.testing.assert_close(targets[0,ids], torch.tensor([
        .6,.1,.08024,.08024*.9487,.2,.2*1.0843,.3,.3*1.0843,.4,.4*1.0843,.5,.5*1.0843]))
    torch.testing.assert_close(term.processed_actions[1], torch.tensor([0.,1.4381,1.4381,1.4381,1.,1.1641]))
    # Full closure agrees with the existing 12-joint calibrated targets.
    term.process_actions(torch.full((3,6),10.))
    term.apply_actions()
    torch.testing.assert_close(targets[:,ids], torch.tensor(robot_module.CLOSED_POSITIONS).expand(3,-1))
    before = targets.clone()
    term.reset(torch.tensor([1]))
    assert targets[1].count_nonzero() == term.raw_actions[1].count_nonzero() == 0
    torch.testing.assert_close(targets[[0,2]],before[[0,2]])
    term.reset()
    assert targets.count_nonzero() == 0
    obs = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.sweep_simple')
    drive_names = importlib.import_module('_baseline_cpu_pkg.assets.hand_kinematics').HAND_DRIVE_JOINT_NAMES
    selected = [articulation_names.index(n) for n in drive_names]
    torch.testing.assert_close(obs.hand_drive_positions(env,NS(name='robot',joint_ids=selected)),robot.data.joint_pos[:,selected])
    last = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.final').previous_action_command
    env.action_manager = NS(action=torch.arange(36.).reshape(3,12)-18)
    torch.testing.assert_close(last(env), env.action_manager.action.tanh())


def test_v0_test_only_changes_last_action_and_fixed_arm_velocity_termination(family, load_mdp):
    parent = family.SweepSimpleFingerEnvCfg()
    before = parent.to_dict()
    cls = family.SweepSimpleFingerV0TestEnvCfg
    assert cls.__bases__ == (family.SweepSimpleFingerEnvCfg,)
    child = cls()
    expected = copy.deepcopy(before)
    expected['terminations']['arm_joint_velocity']['params']['threshold'] = 1.0
    expected['curriculum']['arm_joint_velocity_threshold'] = None
    raw_last = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.sweep_simple').raw_previous_action
    assert child.observations.policy.actions.func is raw_last
    expected['observations']['policy']['actions']['func'] = f'{raw_last.__module__}:{raw_last.__name__}'
    assert_ordered_equal(child.to_dict(), expected)
    # Exercise the actual termination function: strict absolute boundary,
    # selected arm joints only (the last column represents an excluded hand joint).
    fn = importlib.import_module('_baseline_cpu_pkg.mdp.termination_functions.sweep').arm_velocity
    env = NS(scene={'robot': NS(data=NS(joint_vel=torch.tensor([
        [1., -1., 10.], [1.001, 0., 0.], [0., -1.001, 0.], [0., 0., 0.],
    ])))})
    term = child.terminations.arm_joint_velocity
    assert term.params['asset_cfg'].joint_names == parent.terminations.arm_joint_velocity.params['asset_cfg'].joint_names
    result = fn(env, asset_cfg=NS(name='robot', joint_ids=[0, 1]), threshold=term.params['threshold'])
    assert result.tolist() == [False, True, True, False]
    child.terminations.arm_joint_velocity.params['threshold'] = 99.
    assert_ordered_equal(parent.to_dict(), before)
    assert_ordered_equal(family.SweepSimpleFingerEnvCfg().to_dict(), before)


@pytest.mark.parametrize('config_name', ['SweepSimpleFingerV0TestEnvCfg', 'SweepSimpleProfV2EnvCfg'])
def test_runner_wrapper_bounds_history_and_rate_before_osc_scale(family, config_name):
    """Execute upstream wrapper/manager/scaling methods; simulator I/O is a double."""
    def method(path, class_name, name):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        ns = dict(torch=torch, TensorDict=TensorDict)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        return ns[name]

    step = method(PROJECT_ROOT/'IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py',
                  'RslRlVecEnvWrapper', 'step')
    process = method(PROJECT_ROOT/'IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py',
                     'ActionManager', 'process_action')
    scale = method(PROJECT_ROOT/'IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py',
                   'OperationalSpaceControllerAction', '_preprocess_actions')
    cfg = getattr(family, config_name)()
    observe = cfg.observations.policy.actions.func
    cost = cfg.rewards.arm_action_rate.func
    arm = NS(_raw_actions=torch.zeros(2, 6), _processed_actions=torch.zeros(2, 6),
             _pose_abs_idx=None, _pose_rel_idx=0, _wrench_abs_idx=None,
             _stiffness_idx=None, _damping_ratio_idx=None,
             _position_scale=.03, _orientation_scale=.15, action_dim=6)
    arm.process_actions = lambda action: scale(arm, action)
    manager = NS(_action=torch.zeros(2, 12), _prev_action=torch.zeros(2, 12),
                 total_action_dim=12, device='cpu', _terms={'arm': arm})
    manager.action, manager.prev_action = manager._action, manager._prev_action
    env = NS(action_manager=manager, cfg=NS(is_finite_horizon=True))

    def env_step(action):
        process(manager, action)
        reward = -.03 * .02 * (cost(env, 'arm') + cost(env, 'hand'))
        return {'policy': observe(env)}, reward, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

    env.step = env_step
    wrapper = NS(env=env, unwrapped=env, num_envs=2, clip_actions=math.pi/2)
    # Include values inside the boundary, exactly on it and the diagnosed ~9045 outlier.
    raw = torch.tensor([[-9045., -math.pi/2, -1., 0., 1., math.pi/2, 9045., -9045., .1, -.2, 2., -2.]])
    raw = torch.cat((raw, -raw))
    original = raw.clone()
    first = raw.clamp(-math.pi/2, math.pi/2)
    obs, _, _, _ = step(wrapper, raw)
    torch.testing.assert_close(raw, original)  # PPO's original sample is not modified.
    torch.testing.assert_close(obs['policy'], first, rtol=0, atol=0)
    assert obs['policy'][0, 1].item() == pytest.approx(-math.pi/2)  # No tanh.
    torch.testing.assert_close(arm._raw_actions, first[:, :6])
    torch.testing.assert_close(arm._processed_actions[:, :3], first[:, :3] * .03)
    torch.testing.assert_close(arm._processed_actions[:, 3:], first[:, 3:6] * .15)
    obs, reward, _, _ = step(wrapper, -raw)
    torch.testing.assert_close(manager.prev_action, first)
    torch.testing.assert_close(obs['policy'], -first)
    torch.testing.assert_close(reward, -.0006 * (-first-first).square().sum(-1))
    assert reward.abs().max() <= .0006 * 12 * math.pi**2
    # The observation function itself does not silently clamp or squash its input.
    manager.action[:] = 3.
    assert observe(env) is manager.action
    assert (observe(env) == 3.).all()


def test_all_six_hand_channels_cost_without_output_cap(load_mdp):
    fn = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_simple').finger_action_rate_l2
    current = torch.zeros(7,12)
    current[:6,6:] = torch.eye(6)*2
    current[6,0] = 20
    env = NS(action_manager=NS(action=current,prev_action=torch.zeros_like(current)))
    torch.testing.assert_close(fn(env,'hand'),torch.tensor([4.]*6+[0.]))
    torch.testing.assert_close(fn(env,'arm'),torch.tensor([0.]*6+[400.]))
    with pytest.raises(ValueError): fn(env,'invalid')


def test_runtime_reset_clears_six_channels_only_selected_rows():
    class Parent:
        def _reset_idx(self,ids):
            self.called = ids
            self.action_manager._action[ids,-2:] = 1
            self.action_manager._prev_action[ids,-2:] = 1
    ns = _execute_definitions(MODULE_ROOT/'runtime/sweep_simple.py', {'SweepSimpleEnv':Parent}, {'SweepSimpleFingerEnv'})
    env = ns.SweepSimpleFingerEnv()
    env.action_manager = NS(_action=torch.full((3,12),2.),_prev_action=torch.full((3,12),3.))
    env._reset_idx(torch.tensor([1]))
    assert env.called.tolist() == [1]
    assert env.action_manager._action[1,6:].count_nonzero() == 0
    assert env.action_manager._prev_action[1,6:].count_nonzero() == 0
    assert (env.action_manager._action[[0,2]] == 2).all()
    assert (env.action_manager._prev_action[[0,2]] == 3).all()
    assert (env.action_manager._action[1,:6] == 2).all()


def test_registration_and_actual_model_gradients_export(load_mdp):
    entries = {}
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym':NS(registry=entries,register=lambda **kw:entries.update({kw['id']:kw}))})
    entry = entries['Inspire-Sweep-Simple-Finger-v0']
    assert entry['entry_point'].endswith(':SweepSimpleFingerEnv')
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':SweepSimpleFingerEnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepSimpleFingerPPORunnerCfg')
    test_entry = entries['Inspire-Sweep-Simple-Finger-v0-Test']
    assert test_entry['entry_point'] == entry['entry_point']
    assert test_entry['kwargs']['env_cfg_entry_point'].endswith(':SweepSimpleFingerV0TestEnvCfg')
    assert test_entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':SweepSimpleFingerV0TestPPORunnerCfg')
    module = importlib.import_module('_baseline_cpu_pkg.agent.sweep_simple_models')
    cls = module.SweepSimpleFingerModel
    obs = TensorDict({'policy':torch.randn(8,66),'obstacles':torch.randn(8,64)},batch_size=[8])
    obs['obstacles'][:,::4] = 1
    groups = {'actor':['policy','obstacles'],'critic':['policy','obstacles']}
    actor = cls(obs,groups,'actor',12,distribution_cfg={'class_name':'GaussianDistribution','init_std':1.})
    critic = cls(obs,groups,'critic',1)
    for model, width in ((actor,12),(critic,1)):
        result = model(obs)
        assert result.shape == (8,width)
        assert model.get_latent(obs).shape == (8,82)
        result.square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum() > 0
    export = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'],obs['obstacles']),-1)),actor(obs))
    assert actor.as_onnx().get_dummy_inputs()[0].shape == (1,130)


def test_live_action_and_runner_configuration(omni_config, load_mdp):
    from dataclasses import MISSING
    from test_omni_configuration import _AssetNode
    cc = omni_config.baseline.configclass
    @cc
    class Cfg:
        asset_name: str = 'robot'
        distribution_cfg: object = None
        policy: object = None
    class Distribution(_AssetNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    Cfg.GaussianDistributionCfg = Distribution
    hand = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.hand_openness').HandOpennessAction
    finger = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.sweep_simple').FingerJointPositionAction
    # OSC is an unchanged config leaf; use a configclass allowing its full kwargs.
    @cc
    class OscCfg:
        asset_name: str = ''
        joint_names: object = None
        body_name: str = ''
        task_frame_rel_path: object = None
        controller_cfg: object = None
        position_scale: float = 0.
        orientation_scale: float = 0.
    ns = dict(configclass=cc, ActionTermCfg=Cfg, OperationalSpaceControllerActionCfg=OscCfg,
              OperationalSpaceControllerCfg=_AssetNode, WrenchRefactorOSCAction=object,
              HandOpennessAction=hand, FingerJointPositionAction=finger,
              BoundedFingerJointPositionAction=type("BoundedHand", (), {}), BoundedFingerOSCAction=type("BoundedOSC", (), {}),
              SensorFrameWrenchOSCAction=type("SensorWrenchOSC", (), {}),
              ARM_JOINT_NAMES=omni_config.baseline.ARM_JOINT_NAMES)
    actions = _execute_definitions(MODULE_ROOT/'mdp/actions/sweep_simple.py',ns)
    base, child = actions.SweepSimpleActionsCfg(), actions.SweepSimpleFingerActionsCfg()
    assert_ordered_equal(child.arm_action.to_dict(),base.arm_action.to_dict())
    assert base.gripper_action.class_type is hand
    assert child.gripper_action.class_type is finger
    assert child.gripper_action.asset_name == 'robot'
    bounded = actions.SweepSimpleFingerV1ActionsCfg()
    expected = child.to_dict()
    assert bounded.arm_action.class_type is ns['BoundedFingerOSCAction']
    expected['arm_action']['class_type'] = bounded.to_dict()['arm_action']['class_type']
    assert bounded.gripper_action.class_type is ns['BoundedFingerJointPositionAction']
    expected['gripper_action']['class_type'] = bounded.to_dict()['gripper_action']['class_type']
    assert_ordered_equal(bounded.to_dict(), expected)
    assert actions.SweepSimpleFingerActionsCfg().gripper_action.class_type is finger
    wrench = actions.SweepSimpleFingerWrenchActionsCfg()
    expected_hand = child.gripper_action.to_dict()
    expected_hand['class_type'] = wrench.gripper_action.to_dict()['class_type']
    assert wrench.gripper_action.class_type is ns['BoundedFingerJointPositionAction']
    assert_ordered_equal(wrench.gripper_action.to_dict(), expected_hand)
    arm = wrench.arm_action
    assert arm.class_type is ns['SensorFrameWrenchOSCAction']
    assert arm.body_name == 'axia80_link' and arm.body_offset is None
    assert arm.task_frame_rel_path == 'Robot/axia80_link'
    assert arm.wrench_scale == (10., 10., 10., 1., 1., 1.)
    assert arm.effort_limit_scale == child.arm_action.effort_limit_scale == 1.
    assert arm.controller_cfg.target_types == ['wrench_abs']
    assert arm.controller_cfg.motion_control_axes_task == (0,)*6
    assert arm.controller_cfg.contact_wrench_control_axes_task == (1,)*6
    assert arm.controller_cfg.gravity_compensation
    assert not arm.controller_cfg.inertial_dynamics_decoupling
    assert arm.controller_cfg.contact_wrench_stiffness_task is None
    assert arm.controller_cfg.nullspace_control == arm.nullspace_joint_pos_target == 'none'
    assert_ordered_equal(actions.SweepSimpleFingerActionsCfg().to_dict(), child.to_dict())

    runner = _execute_definitions(MODULE_ROOT/'agent/sweep_simple_ppo_cfg.py', dict(
        configclass=cc, MISSING=MISSING, RslRlMLPModelCfg=Cfg,
        RslRlOnPolicyRunnerCfg=Cfg, RslRlPpoAlgorithmCfg=_AssetNode))
    parent = runner.SweepSimpleV1PPORunnerCfg().to_dict()
    actual = runner.SweepSimpleFingerPPORunnerCfg().to_dict()
    expected = copy.deepcopy(parent)
    expected['experiment_name'] = 'inspire_sweep_simple_finger_v0'
    for name in ('actor','critic'):
        expected[name]['class_name'] = 'inspire_sweep_baseline.agent.sweep_simple_models:SweepSimpleFingerModel'
    assert_ordered_equal(actual,expected)
    assert_ordered_equal(runner.SweepSimpleV1PPORunnerCfg().to_dict(),parent)
    test_expected = copy.deepcopy(actual)
    test_expected['experiment_name'] = 'inspire_sweep_simple_finger_v0_test'
    test_expected['clip_actions'] = math.pi / 2
    assert runner.SweepSimpleFingerV0TestPPORunnerCfg.__bases__ == (runner.SweepSimpleFingerPPORunnerCfg,)
    assert_ordered_equal(runner.SweepSimpleFingerV0TestPPORunnerCfg().to_dict(), test_expected)
    assert_ordered_equal(runner.SweepSimpleFingerPPORunnerCfg().to_dict(), actual)
    wrench_expected = copy.deepcopy(actual)
    wrench_expected['experiment_name'] = 'inspire_sweep_simple_finger_v0_wrench'
    assert runner.SweepSimpleFingerWrenchPPORunnerCfg.__bases__ == (runner.SweepSimpleFingerPPORunnerCfg,)
    assert_ordered_equal(runner.SweepSimpleFingerWrenchPPORunnerCfg().to_dict(), wrench_expected)
