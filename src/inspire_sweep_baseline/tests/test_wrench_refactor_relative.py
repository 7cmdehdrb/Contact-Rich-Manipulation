"""Relative-frame invariance, snapshot isolation, tactile states and shared PPO storage."""
import importlib
import importlib.util
import math
from types import SimpleNamespace as NS

import torch
from tensordict import TensorDict

from conftest import MODULE_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config


def test_uncapped_raw_action_rate_preserves_group_semantics(configs):
    func = configs('reward_functions', 'wrench_refactor').policy_action_rate_penalty
    action = torch.tensor([[3., 4., 0., 0., 0., 0., 1., -2.],
                           [6., 8., 0., 0., 0., 0., 1., 1.],
                           [8., 8., 0., 0., 0., 0., 40., 73.],
                           [12000., -9000., 0., 0., 0., 0., -40000., 40000.]])
    env = NS(action_manager=NS(action=action, prev_action=torch.zeros_like(action)))
    original = action.clone()
    cost = func(env)
    expected = .03 * action[:, :6].square().sum(-1) + .03 * action[:, -2:].clamp(0., 1.).square().sum(-1)
    torch.testing.assert_close(cost, expected)
    assert cost[-1] > 1e6
    assert torch.equal(action, original)


def test_runtime_installs_final_clip_without_changing_reader_or_saved_cfg():
    import ast
    import copy
    tree = ast.parse((MODULE_ROOT / 'runtime/wrench_refactor.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    reader = lambda env: torch.tensor([[900., -500., 100., 20., -30., 1.]])
    source = NS(policy=NS(wrist_wrench=NS(func=reader, clip=None)),
                critic=NS(wrist_wrench=NS(func=reader, clip=None)))
    class Base:
        def load_managers(self):
            self.observation_manager = NS(cfg=copy.deepcopy(source),
                active_terms={'policy':['wrist_wrench'], 'critic':['wrist_wrench']})
    namespace = {'torch':torch, 'ManagerBasedRLEnv':Base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<runtime>', 'exec'), namespace)
    env = namespace['WrenchRefactorEnv'].__new__(namespace['WrenchRefactorEnv'])
    env.scene = {'shelf':NS(data=NS(root_pos_w=torch.zeros(1,3), root_quat_w=torch.zeros(1,4)))}
    env.cfg = NS(wrist_wrench_observation_limits=(150.,150.,150.,8.,8.,8.))
    env.device = 'cpu'
    env.load_managers()
    for name in ('policy','critic'):
        term = getattr(env.observation_manager.cfg,name).wrist_wrench
        assert term.func is reader
        raw = term.func(env)
        assert raw[0,0] == 900  # Compensation/telemetry source is NOT saturated.
        torch.testing.assert_close(raw.clip(min=term.clip[0],max=term.clip[1]),
                                   torch.tensor([[150.,-150.,100.,8.,-8.,1.]]))
        assert getattr(source,name).wrist_wrench.clip is None
    env.cfg = NS()  # Non-Relative runtime leaves the parent observation unchanged.
    env.load_managers()
    assert env.observation_manager.cfg.policy.wrist_wrench.clip is None


def relative_env():
    q = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    root = NS(root_pos_w=torch.tensor([[3., 2., 1.], [-3., 1., 2.]]), root_quat_w=q.clone())
    eef = NS(target_pos_w=(root.root_pos_w + torch.tensor([1., 0., 0.]))[:, None],
             target_quat_w=q[:, None].clone())
    command = NS(target_init_state_w=torch.cat((root.root_pos_w + torch.tensor([2., 1., 0.]), q), -1),
                 command=root.root_pos_w + torch.tensor([3., 2., 0.]))
    env = NS(num_envs=2, device='cpu', scene={'robot':NS(data=root), 'ee_frame':NS(data=eef)},
             cfg=NS(observations=NS(policy=NS(enable_corruption=True))),
             command_manager=NS(get_term=lambda name:command, get_command=lambda name:command.command))
    rows = torch.zeros(2, 16, 4)
    rows[:, 0] = torch.tensor([1., 2., 1., 0.])
    rows[:, 1, 1:] = 123.  # hidden slots must stay exactly zero after translation
    env._initial_obstacle_observation = rows.flatten(1)
    return env, command, eef


def test_current_eef_transform_and_no_live_geometry(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    env, command, eef = relative_env()
    snapshot = env._initial_obstacle_observation.clone()
    term = m.InitialObjectPositionInEEF(NS(params={}), env)
    initial = term(env, coordinate_noise=0.)
    torch.testing.assert_close(initial, torch.tensor([[1., 1., 0.]]).repeat(2, 1))
    torch.testing.assert_close(m.goal_position_in_eef(env), torch.tensor([[2., 2., 0.]]).repeat(2, 1))
    eef.target_pos_w[:, 0, 0] += 1.
    eef.target_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    expected = torch.tensor([[1., 0., 0.]]).repeat(2, 1)
    torch.testing.assert_close(term(env), expected, atol=2e-6, rtol=0)
    rows = m.initial_obstacle_positions_in_eef(env).reshape(2, 16, 4)
    torch.testing.assert_close(rows[:, 0, 1:], expected, atol=2e-6, rtol=0)
    assert not rows[:, 1:].any()
    assert torch.equal(snapshot, env._initial_obstacle_observation)
    # Changing even the command's snapshot has no effect until that env resets.
    command.target_init_state_w[:, 0] += 5.
    torch.testing.assert_close(term(env), expected, atol=2e-6, rtol=0)
    term.reset(torch.tensor([0]))
    result = term(env, coordinate_noise=0.)
    torch.testing.assert_close(result[0], torch.tensor([1., -5., 0.]), atol=2e-6, rtol=0)
    torch.testing.assert_close(result[1], expected[1], atol=2e-6, rtol=0)


def test_target_noise_is_fixed_in_root_and_partial_reset_only(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    env, _, eef = relative_env()
    term = m.InitialObjectPositionInEEF(NS(params={}), env)
    torch.manual_seed(41)
    first = term(env)
    bias = first - torch.tensor([1., 1., 0.])
    assert (bias.abs() <= .010001).all() and bias.abs().sum() > 0
    torch.testing.assert_close(first, term(env))
    eef.target_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    rotated = term(env)
    torch.testing.assert_close(rotated, first * torch.tensor([-1., -1., 1.]))
    term.reset(torch.tensor([0]))
    updated = term(env)
    torch.testing.assert_close(updated[1], rotated[1])
    assert not torch.equal(updated[0], rotated[0])


def test_base_obstacles_match_refactor_and_ignore_eef_motion(load_mdp):
    relative = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    base = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.revision')
    env, _, eef = relative_env()
    snapshot = env._initial_obstacle_observation.clone()
    first = base.initial_obstacle_positions(env)
    before_relative = relative.initial_obstacle_positions_in_eef(env)
    eef.target_pos_w += 3.
    eef.target_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    assert torch.equal(first, base.initial_obstacle_positions(env))
    assert not torch.equal(before_relative, relative.initial_obstacle_positions_in_eef(env))
    rows = first.reshape(2,16,4)
    assert torch.equal(rows[:,0], torch.tensor([[1.,2.,1.,0.]]).repeat(2,1))
    assert not rows[:,1:].any()
    first.zero_()
    assert torch.equal(snapshot, env._initial_obstacle_observation)


def test_tactile_four_states_and_original_bit_noise(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    bits = torch.zeros(4, 34)
    bits[1, :17] = bits[2, 17:] = bits[3, :] = 1.
    encoded = m.encode_paired_tactile(bits)
    assert torch.equal(encoded, torch.tensor([0., 1., -1., 2.])[:, None].expand(-1, 17))
    torch.manual_seed(77)
    expected = m.encode_paired_tactile(m.tactile_observation_noise(bits, None))
    torch.manual_seed(77)
    assert torch.equal(m.paired_tactile_observation_noise(encoded, None), expected)


def test_observations_only_expose_shared_actor_inputs(configs, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    module = configs('observations', 'wrench_refactor')
    parent = module.WrenchRefactorObservationsCfg()
    before = parent.to_dict()
    relative = module.WrenchRefactorRelativeObservationsCfg()
    assert relative.critic is relative.critic_obstacles is None
    assert parent.to_dict() == before
    assert list(vars(parent.policy)) == list(vars(relative.policy))
    assert relative.policy.ee_pose.to_dict() == parent.policy.ee_pose.to_dict()
    assert relative.policy.wrist_wrench.to_dict() == parent.policy.wrist_wrench.to_dict()


def test_shared_models_storage_gradients_and_export():
    from rsl_rl.storage import RolloutStorage
    spec = importlib.util.spec_from_file_location('_relative_models', MODULE_ROOT / 'agent/wrench_refactor_models.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = torch.randn(8, 16, 4); rows[..., 0] = 1.
    obs = TensorDict({'policy':torch.randn(8, 58), 'obstacles':rows.flatten(1)}, batch_size=[8])
    groups = {'actor':['policy','obstacles'], 'critic':['policy','obstacles']}
    actor = m.WrenchRefactorRelativeModel(obs, groups, 'actor', 8,
        distribution_cfg={'class_name':'GaussianDistribution', 'init_std':1.})
    critic = m.WrenchRefactorRelativeModel(obs, groups, 'critic', 1, obstacle_group='obstacles')
    assert actor.obs_dim == critic.obs_dim == 58
    for model in (actor, critic):
        model(obs).square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum() > 0
    export = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'], obs['obstacles']), -1)), actor(obs))
    storage = RolloutStorage('rl', 8, 2, obs, [8], 'cpu')
    assert set(storage.observations.keys()) == {'policy', 'obstacles'}
    assert storage.observations['policy'].shape == (2, 8, 58)


def test_v2_live_target_preserves_bias_and_eef_transform(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    env, command, eef = relative_env()
    positions = command.target_init_state_w[:, None, :3].repeat(1, 2, 1)
    positions[:, 1] += 20.  # The distractor must never become the target.
    env.scene['object_collection'] = NS(data=NS(object_pos_w=positions))
    env.target_id = torch.zeros(2, 1, dtype=torch.long)
    critic = m.CurrentObjectPositionInEEF(NS(params={}), env)
    actor = m.InitialObjectPositionInEEF(NS(params={}), env)
    torch.manual_seed(41)
    first = critic(env)
    torch.manual_seed(41)
    torch.testing.assert_close(first, actor(env))
    positions[:, 0, 0] += .3
    torch.testing.assert_close(critic(env), first + torch.tensor([.3, 0., 0.]))
    torch.testing.assert_close(actor(env), first)
    eef.target_pos_w[:, 0, 0] += .2
    eef.target_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    rotated = (first + torch.tensor([.1, 0., 0.])) * torch.tensor([-1., -1., 1.])
    torch.testing.assert_close(critic(env), rotated)
    critic.reset(torch.tensor([0]))
    updated = critic(env)
    torch.testing.assert_close(updated[1], rotated[1])
    assert not torch.equal(updated[0], rotated[0])
    env.target_id[0, 0] = 1
    env.cfg.observations.policy.enable_corruption = False
    expected = (positions[torch.arange(2), env.target_id[:, 0]] - eef.target_pos_w[:, 0])
    expected *= torch.tensor([-1., -1., 1.])
    torch.testing.assert_close(critic(env), expected)


def test_v2_observation_config_only_replaces_critic_target(configs, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', importlib.import_module('_omni_official_utils.noise'))
    module = configs('observations', 'wrench_refactor')
    parent = module.WrenchRefactorRelativeObservationsCfg()
    before = parent.to_dict()
    child = module.WrenchRefactorRelativeV2ObservationsCfg()
    expected = parent.to_dict()
    expected['critic'] = parent.policy.to_dict()
    expected['critic']['target_obs_state']['func'] = child.critic.target_obs_state.to_dict()['func']
    assert child.to_dict() == expected
    assert parent.to_dict() == before
    assert child.policy.target_obs_state.func.__name__ == 'InitialObjectPositionInEEF'
    assert child.critic.target_obs_state.func.__name__ == 'CurrentObjectPositionInEEF'
    assert child.policy is not child.critic


def test_v2_network_input_isolation_storage_and_export():
    from rsl_rl.storage import RolloutStorage
    spec = importlib.util.spec_from_file_location('_relative_v2_models', MODULE_ROOT / 'agent/wrench_refactor_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = torch.randn(8, 16, 4)
    rows[..., 0] = 1.
    obs = TensorDict({'policy': torch.randn(8, 58), 'critic': torch.randn(8, 58),
                     'obstacles': rows.flatten(1)}, batch_size=[8])
    groups = {'actor': ['policy', 'obstacles'], 'critic': ['critic', 'obstacles']}
    actor = module.WrenchRefactorRelativeV2Model(obs, groups, 'actor', 8,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.})
    critic = module.WrenchRefactorRelativeV2Model(obs, groups, 'critic', 1)
    for model, own_group, other_group in ((actor, 'policy', 'critic'), (critic, 'critic', 'policy')):
        before = model(obs)
        changed = obs.clone()
        changed[other_group][:, 21:24] += 10.
        torch.testing.assert_close(model(changed), before)
        changed[own_group][:, 21:24] += 10.
        assert not torch.allclose(model(changed), before)
        model(obs).square().mean().backward()
        assert model.obstacle_encoder.mlp[0].weight.grad.abs().sum() > 0
    export = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(export(torch.cat((obs['policy'], obs['obstacles']), -1)), actor(obs))
    storage = RolloutStorage('rl', 8, 2, obs, [8], 'cpu')
    assert set(storage.observations.keys()) == {'policy', 'critic', 'obstacles'}
    assert storage.observations['policy'].shape == storage.observations['critic'].shape == (2, 8, 58)
    assert storage.observations['obstacles'].shape == (2, 8, 64)


def test_v2_runner_preserves_ppo_settings(omni_config):
    import ast
    from dataclasses import MISSING
    from typing import Literal
    from conftest import PROJECT_ROOT
    from test_omni_configuration import _execute_definitions
    official_path = PROJECT_ROOT / 'IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py'
    classes = {node.name for node in ast.parse(official_path.read_text()).body if isinstance(node, ast.ClassDef)}
    official = _execute_definitions(official_path, dict(configclass=omni_config.baseline.configclass,
        MISSING=MISSING, Literal=Literal), classes)
    module = _execute_definitions(MODULE_ROOT / 'agent/wrench_refactor_ppo_cfg.py', vars(official).copy())
    parent = module.WrenchRefactorRelativePPORunnerCfg()
    before = parent.to_dict()
    child = module.WrenchRefactorRelativeV2PPORunnerCfg()
    expected = parent.to_dict()
    expected['experiment_name'] = 'inspire_wrench_refactor_relative_v2'
    expected['obs_groups']['critic'] = ['critic', 'obstacles']
    for group in ('actor', 'critic'):
        expected[group]['class_name'] = 'inspire_sweep_baseline.agent.wrench_refactor_models:WrenchRefactorRelativeV2Model'
    assert child.to_dict() == expected
    assert parent.to_dict() == before
    variant = module.WrenchRefactorRelativeV2_1PPORunnerCfg()
    expected_variant = child.to_dict()
    expected_variant['experiment_name'] = 'inspire_wrench_refactor_relative_v2_1'
    assert variant.to_dict() == expected_variant


def test_relative_velocity_curriculum_boundaries_and_rewind(load_mdp):
    m = importlib.import_module('_baseline_cpu_pkg.mdp.curriculum_functions.wrench_refactor')
    term = NS(params={'threshold': 4., 'asset_cfg': 'arm-only'})
    manager = NS(get_term_cfg=lambda name: term,
                 set_term_cfg=lambda name, value: None)
    env = NS(learning_iteration=0, termination_manager=manager)
    # Absolute schedules must also support checkpoint jumps, reset and rewind.
    for iteration, expected in [(0,4.), (9999,4.), (10000,4.), (10249,4.),
                                (10250,3.95), (10499,3.95), (10500,3.9),
                                (14250,3.15), (14500,3.1), (19750,2.05),
                                (19999,2.05), (20000,2.), (90000,2.), (0,4.)]:
        env.learning_iteration = iteration
        assert m.arm_velocity_threshold_schedule(env, None) == expected
        assert term.params == {'threshold': expected, 'asset_cfg': 'arm-only'}
        assert m.arm_velocity_threshold_schedule(env, torch.tensor([0])) == expected


def test_relative_velocity_curriculum_updates_real_termination(load_mdp):
    schedule = importlib.import_module('_baseline_cpu_pkg.mdp.curriculum_functions.wrench_refactor')
    functions = importlib.import_module('_baseline_cpu_pkg.mdp.termination_functions.sweep')
    cfg = NS(name='robot', joint_ids=[0, 2])
    velocity = torch.tensor([[4., 99., -4.], [4.001, 0., 0.], [2., 0., -2.], [2.001, 0., 0.]])
    term = NS(params={'asset_cfg': cfg, 'threshold': 4.})
    env = NS(learning_iteration=0, scene={'robot': NS(data=NS(joint_vel=velocity))},
             termination_manager=NS(get_term_cfg=lambda name: term, set_term_cfg=lambda name, value: None))
    schedule.arm_velocity_threshold_schedule(env, None)
    assert functions.arm_velocity(env, **term.params).tolist() == [False, True, False, False]
    env.learning_iteration = 20000
    schedule.arm_velocity_threshold_schedule(env, None)
    assert functions.arm_velocity(env, **term.params).tolist() == [True, True, False, True]


def motion_reward_env(count=1):
    from test_wrench_refactor import reward_env
    env, pos, forces = reward_env(count)
    env.step_dt = .02
    env.cfg.contact_height_limit = env.cfg.goal_reward_height_limit = .14
    env.scene['ee_frame'].data.target_pos_w[..., 2] = 1.18
    pos[:, 0, :2] = 0.
    goal = torch.zeros(count, 3, dtype=pos.dtype)
    goal[:, 0] = .1
    env.command_manager.get_command = lambda _: goal
    return env, pos, forces, goal


def opposite_reach_env(count=1):
    env, pos, _, goal = motion_reward_env(count)
    # Match the simulator's float32 state and the parent reward history buffers.
    pos, goal = pos.float(), goal.float()
    env.scene['object_collection'].data.object_pos_w = pos
    env.scene['ee_frame'].data.target_pos_w = env.scene['ee_frame'].data.target_pos_w.float()
    env.command_manager.get_command = lambda _: goal
    env.scene['robot'] = NS(data=NS())
    env._revision_bounds_t = torch.zeros(count, 1, 2, 3)
    env._revision_bounds_t[:, 0, 0, 2] = -.02
    env._revision_bounds_t[:, 0, 1, 2] = .10
    env.sweep_dir = goal - pos[:, 0]
    return env, pos, goal


def test_opposite_reaching_point_xy_height_live_and_coincident(configs):
    m = configs('reward_functions', 'wrench_refactor')
    env, pos, goal = opposite_reach_env(5)
    goal[:, :2] = torch.tensor([[.1, 0.], [-.1, 0.], [0., .1], [0., -.1], [.1, .1]])
    env.sweep_dir = goal - pos[:, 0]
    point = m.opposite_reaching_point(env)
    delta = point[:, :2] - pos[:, 0, :2]
    torch.testing.assert_close(delta.norm(dim=-1), torch.full((5,), .05))
    torch.testing.assert_close(delta, -.05 * goal[:, :2] / goal[:, :2].norm(dim=-1, keepdim=True))
    torch.testing.assert_close(point[:, 2], torch.full((5,), 1.11))
    # The physical object's height is scale-correct; shelf floor, not link origin, sets Z.
    env._revision_bounds_t[0, 0, 1, 2] = .22
    pos[0, 0] = torch.tensor([.02, .03, 1.4])
    moved = m.opposite_reaching_point(env)
    assert abs(moved[0, 2].item() - 1.17) < 1e-6
    torch.testing.assert_close((moved[0, :2] - pos[0, 0, :2]).norm(), torch.tensor(.05))
    pos[1, 0, :2] = goal[1, :2]
    coincident = m.opposite_reaching_point(env)
    torch.testing.assert_close(coincident[1, :2], torch.tensor([-.05, 0.]))
    assert torch.isfinite(coincident).all()


def test_opposite_reach_reset_first_step_budget_partial_and_dt(configs):
    m = configs('reward_functions', 'wrench_refactor')
    env, pos, goal = opposite_reach_env(2)
    eef = env.scene['ee_frame'].data.target_pos_w
    eef[:, 0] = torch.tensor([-.35, 0., 1.11])
    cfg = NS(params={'near_radius': .10, 'offset': .05})
    # Command still belongs to the previous episode when reward reset runs.
    goal[:, 0] = -10.
    term = m.OppositeReachProgress(cfg, env)
    goal[:, 0] = .1
    torch.testing.assert_close(term(env), torch.zeros(2), atol=1e-6, rtol=0.)
    eef[:, 0, 0] += .04
    torch.testing.assert_close(term(env) * env.step_dt, torch.full((2,), .04))
    assert not term(env).any()  # Waiting does not pay twice.
    eef[:, 0, 0] -= .02
    assert not term(env).any()
    eef[:, 0, 0] += .02
    assert not term(env).any()
    preserved = term._best_distance[1].clone()
    eef[0, 0, 0] = -.5
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term._best_distance[1], preserved)
    env.step_dt = .01
    eef[0, 0, 0] += .03
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([.03, 0.]))


def test_opposite_reach_near_radius_boundary(configs):
    m = configs('reward_functions', 'wrench_refactor')
    env, _, _ = opposite_reach_env(3)
    eef = env.scene['ee_frame'].data.target_pos_w
    eef[:, 0] = torch.tensor([-.25, 0., 1.11])
    term = m.OppositeReachProgress(NS(params={'near_radius': .05}), env)
    eef[:, 0, 0] = torch.tensor([-.100001, -.10, -.099999])
    paid = term(env, near_radius=.05) * env.step_dt
    torch.testing.assert_close(paid, torch.tensor([.149999, .15, .15]), atol=1e-7, rtol=1e-5)
    assert not term(env, near_radius=.05).any()


def test_relative_motion_anchor_threshold_and_per_step_evaluation(configs):
    m = configs('reward_functions', 'wrench_refactor')
    env, pos, _, goal = motion_reward_env(3)
    term = m.RelativeMotionCosineGoalDistance(NS(params={}), env)
    assert not term(env).any()
    env.common_step_counter += 1
    pos[:, 0, 0] = pos.new_tensor([.000999, .001, .001001])
    result = term(env)
    assert (result > 0).all()  # 1 mm is NOT a minimum reward displacement.
    assert term._anchor[0, 0] == 0.
    torch.testing.assert_close(term._anchor[1:, 0], pos[1:, 0, 0])
    torch.testing.assert_close(term(env), result)  # Updating anchor cannot change duplicate calls.
    env.common_step_counter += 1
    pos[:, 0, 0] += .0002
    assert (term(env) > 0).all()
    torch.testing.assert_close(term._anchor[0], pos[0, 0, :2])
    # A sub-mm retreat still evaluates against retained anchor, as requested.
    env.common_step_counter += 1
    pos[:, 0, 0] -= .0001
    assert term(env).tolist()[0] < 0 < term(env).tolist()[1]
    # Orthogonal motion, overshoot, and returning after overshoot.
    env.common_step_counter += 1
    pos[0, 0, :2] = torch.tensor([.12, 0.])
    assert term(env)[0] < 0
    env.common_step_counter += 1
    pos[0, 0, 0] = .11
    assert term(env)[0] > 0
    env.common_step_counter += 1
    pos[0, 0, :2] = goal[0, :2]
    assert term(env)[0] == 0  # Goal vector has no direction at exact goal.
    env.common_step_counter += 1
    pos[0, 0, :2] = torch.tensor([.04, 0.])
    term.reset(torch.tensor([0]))
    pos[0, 0, 1] = .002
    goal[0, :2] = pos.new_tensor([.1, .002])
    assert term(env)[0] == 0


def test_relative_inverse_distance_reset_dt_and_no_contact_gate(configs):
    m = configs('reward_functions', 'wrench_refactor')
    env, pos, forces, goal = motion_reward_env(2)
    term = m.RelativeInverseDistance(NS(params={}), env)
    forces.zero_()
    pos[:, 0, 0] = torch.tensor([.01, -.01])
    torch.testing.assert_close(term(env) * env.step_dt, pos.new_tensor([0., .01]))
    torch.testing.assert_close(term(env) * env.step_dt, pos.new_tensor([0., .01]))
    env.common_step_counter += 1
    pos[:, 0, 0] += .001
    assert not term(env).any()
    # Partial reset and a newly sampled goal must not charge a teleport.
    pos[0, 0, 0] = 10.
    term.reset(torch.tensor([0]))
    goal[0, 0] = 10.1
    assert term(env)[0] == 0.
    env.common_step_counter += 1
    pos[:, 0, 0] -= .002
    torch.testing.assert_close(term(env) * env.step_dt, pos.new_tensor([.002, .002]))
    env.step_dt = .01
    env.common_step_counter += 1
    pos[:, 0, 0] -= .002
    torch.testing.assert_close(-12.97 * term(env) * env.step_dt, pos.new_tensor([-.02594, -.02594]))
    term.reset()
    assert not term(env).any()


def test_relative_v3_contact_gate_advances_history_even_without_contact(configs):
    m = configs('rewards', 'wrench_refactor')
    parent = m.WrenchRefactorRelativeRewardsCfg()
    before = parent.to_dict()
    child = m.WrenchRefactorRelativeV3RewardsCfg()
    expected = parent.to_dict()
    expected['target_contact'] = None
    expected['target_goal_distance'] = child.target_goal_distance.to_dict()
    assert child.to_dict() == expected
    assert parent.to_dict() == before
    assert child.target_goal_distance.weight == 25.
    env, pos, forces, _ = motion_reward_env()
    cfg = child.target_goal_distance
    term = cfg.func(cfg, env)
    forces.zero_()
    pos[0, 0, 0] = .02
    assert term(env, **cfg.params) == 0
    assert term._anchor[0, 0] == .02
    env.common_step_counter += 1
    forces[0, 0, 0, 0] = .02
    pos[0, 0, 0] = .019
    assert term(env, **cfg.params) < 0  # New local motion, not initial-to-current.
    env.common_step_counter += 1
    env.sensor_data_fresh[:] = False
    pos[0, 0, 0] = .03
    assert term(env, **cfg.params) == 0


def test_inverse_distance_calibration_and_old_refactor_preservation(configs):
    m = configs('rewards', 'wrench_refactor')
    integral = 5. * 3. * .05 / .1 * (1. - math.exp(-.1 / .05))
    cfg = m.WrenchRefactorRelativeRewardsCfg().target_inverse_distance
    assert cfg.weight == -round(.2 * integral / .1, 2) == -12.97
    assert abs(abs(cfg.weight) * .1 / integral - .2) < 1e-5
    old = m.WrenchRefactorV2RewardsCfg()
    assert old.target_goal_distance.func.__name__ == 'displacement_cosine_goal_distance'
    assert not hasattr(old, 'target_inverse_distance')
