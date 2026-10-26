"""Final v1/v2 numerical, curriculum, geometry and reset regression contracts."""

import ast
import importlib
import math
import runpy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_final_configuration import configs
from test_omni_configuration import omni_config, _execute_definitions
from test_contact_sweep_clutter_mdp import multiobject_env
from test_fusion_holding_v0 import environment


@pytest.fixture
def final_rewards(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.final')


@pytest.fixture
def geometry(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.common.final_geometry')


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('term_name', ['object_size_randomization', 'object_materials'])
def test_final_startup_events_pass_actual_manager_argument_validation(configs, omni_config, version, term_name):
    """Run Isaac Lab's real validator without starting the simulator."""
    import inspect

    tree = ast.parse((PROJECT_ROOT / 'IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py').read_text())
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ManagerBase')
    validator = next(node for node in manager.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_resolve_common_term_cfg')
    namespace = {'inspect': inspect, 'ManagerTermBaseCfg': omni_config.terms.ManagerTermBaseCfg}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), '<actual-manager-validator>', 'exec'), namespace)
    cfg = getattr(configs('events', 'final'), f'FinalV{version}EventsCfg')()
    env = NS(sim=NS(is_playing=lambda: False))
    namespace['_resolve_common_term_cfg'](NS(_env=env), term_name, getattr(cfg, term_name), min_argc=2)


def test_motion_merge_is_numerically_equal_with_separate_caps(final_rewards, make_env):
    env = make_env(8)
    gen = torch.Generator().manual_seed(12)
    env.action_manager.action[:] = torch.randn((8, 8), generator=gen) * 12
    env.action_manager.prev_action[:] = torch.randn((8, 8), generator=gen) * 4
    env.scene['robot'].data.joint_vel[:] = torch.randn((8, 18), generator=gen) * 6
    arm = NS(name='robot', joint_ids=[0, 3, 6, 9, 12, 15])
    hand = NS(name='robot', joint_ids=[i for i in range(18) if i not in arm.joint_ids])
    for dt in (.01, .02, .1):
        expected = sum(-.03 * final_rewards.action_rate_l2(env, action_group=group, max_cost=100.) for group in ('arm', 'hand'))
        torch.testing.assert_close(-final_rewards.action_rate_penalty(env) * dt, expected * dt)
        expected = (-.03 * final_rewards.joint_vel_l2(env, asset_cfg=arm, max_cost=100.)
                    -.01 * final_rewards.joint_vel_l2(env, asset_cfg=hand, max_cost=100.))
        torch.testing.assert_close(-final_rewards.joint_velocity_penalty(env, arm, hand) * dt, expected * dt)


def test_object_merge_keeps_history_caps_and_partial_reset(final_rewards, make_env):
    env = multiobject_env(make_env, count=4)
    data = env.scene['object_collection'].data
    env._initial_object_positions_w = data.object_pos_w.clone()
    env._active_object_mask = torch.ones(data.object_pos_w.shape[:2], dtype=torch.bool)
    env.common_step_counter = env._sim_step_counter = 0
    data.object_com_lin_acc_w = torch.zeros_like(data.object_pos_w)
    cfg = NS(params={})
    merged = final_rewards.ObjectPenalty(cfg, env)
    previous = final_rewards.ObjectAccelerationPenalty(cfg, env)
    for step, weight in enumerate([0., 0., -.01, -.03, -.05, 0., -.05]):
        env.common_step_counter = step
        env._sim_step_counter = step * 2
        data.object_com_lin_acc_w.fill_(30 + step)
        data.object_pos_w[:, 1, 0] += .01
        if step in (2, 5, 6):
            previous.reset()
        if step == 4:
            merged.reset([0])
            previous.reset([0])
        expected = 30 * final_rewards.combined_active_object_cost(env, displacement_ratio=5)
        if weight:
            expected -= weight * previous(env, free_acceleration=20, max_cost=100)
        torch.testing.assert_close(merged(env, free_acceleration=20, acceleration_weight=weight), expected)


def test_action_feedback_bounds_large_commands_without_mutating_raw_actions(load_mdp):
    observation = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.final')
    legacy = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.contact_sweep_clutter_fusion')
    raw = torch.tensor([[1.e20, -1.e20, 2., -2., .5, -.5, -3., 4.],
                        [0., 0., 0., 0., 0., 0., .25, .8],
                        [0., 0., 0., 0., 0., 0., 1., 1.]])
    original = raw.clone()
    env = NS(action_manager=NS(action=raw))
    actual = observation.previous_action_command(env)
    expected = torch.tensor([[1., -1., math.tanh(2.), math.tanh(-2.),
                              math.tanh(.5), math.tanh(-.5), math.tanh(-3.), math.tanh(4.)],
                             [0., 0., 0., 0., 0., 0., math.tanh(.25), math.tanh(.8)],
                             [0., 0., 0., 0., 0., 0., math.tanh(1.), math.tanh(1.)]])
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(legacy.bounded_last_action(env), original.clamp(-1., 1.))
    actual.zero_()
    torch.testing.assert_close(raw, original)


def test_action_feedback_override_is_scoped_to_v1(configs, monkeypatch):
    import sys
    from types import ModuleType
    from test_omni_configuration import _AssetNode

    noise = ModuleType('isaaclab.utils.noise')
    noise.NoiseCfg = noise.ConstantNoiseCfg = noise.NoiseModelWithAdditiveBiasCfg = noise.UniformNoiseCfg = _AssetNode
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.noise', noise)
    observations = configs('observations', 'final')
    assert observations.FinalObservationsCfg().policy.actions.func is observations.bounded_last_action
    assert observations.FinalV1ObservationsCfg().policy.actions.func is observations.previous_action_command
    old, new = observations.FinalObservationsCfg().policy, observations.FinalV1ObservationsCfg().policy
    assert old.tactile.noise is None and old.wrist_wrench.noise is None
    assert new.tactile.func is old.tactile.func and new.tactile.params == old.tactile.params
    assert new.wrist_wrench.func is old.wrist_wrench.func
    assert new.tactile.noise.func is observations.tactile_observation_noise
    assert new.wrist_wrench.noise.func is observations.wrist_wrench_observation_noise


def test_tactile_observation_noise_flips_both_states_independently(load_mdp):
    observation = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.final')
    clean = torch.zeros(65536, 17)
    clean[1::2] = 1.
    original = clean.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        noisy = observation.tactile_observation_noise(clean, None)
        next_sample = observation.tactile_observation_noise(clean, None)
    assert ((noisy == 0) | (noisy == 1)).all()
    flips = (noisy != clean).float()
    torch.testing.assert_close(flips.mean(0), torch.full((17,), .01), rtol=0., atol=.0015)
    assert flips[::2].mean().item() == pytest.approx(.01, abs=.0005)
    assert flips[1::2].mean().item() == pytest.approx(.01, abs=.0005)
    assert not torch.equal(noisy, next_sample)
    noisy.zero_()
    torch.testing.assert_close(clean, original)


def test_wrist_wrench_observation_noise_has_requested_per_axis_sigma(load_mdp):
    observation = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.final')
    clean = torch.tensor([1., -2., 3., -.1, .2, -.3]).expand(65536, -1).clone()
    original = clean.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        noisy = observation.wrist_wrench_observation_noise(clean, None)
        next_sample = observation.wrist_wrench_observation_noise(clean, None)
    delta = noisy - clean
    torch.testing.assert_close(delta.mean(0), torch.zeros(6), rtol=0., atol=.0007)
    torch.testing.assert_close(delta.std(0), torch.tensor([.05] * 3 + [.005] * 3), rtol=.015, atol=0.)
    correlation = torch.corrcoef(delta.T) - torch.eye(6)
    assert correlation.abs().max() < .02
    assert not torch.equal(noisy, next_sample)
    noisy.zero_()
    torch.testing.assert_close(clean, original)


def test_hand_configuration_inverts_actual_joint_targets(load_mdp):
    observation = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.final')
    action = importlib.import_module('_baseline_cpu_pkg.mdp.action_functions.hand_openness')
    openness = torch.tensor([[0., 0.], [1., 1.], [.25, .8], [1.5, -.5]])
    q = action.hand_joint_targets(openness, torch.tensor(observation.OPEN_POSITIONS), torch.tensor(observation.CLOSED_POSITIONS))
    env = NS(scene={'robot': NS(data=NS(joint_pos=q))})
    actual = observation.hand_configuration(env, NS(name='robot', joint_ids=slice(None)))
    torch.testing.assert_close(actual, openness.clamp(0, 1))


def test_timeout_is_disjoint_exhaustive_and_never_early(configs, monkeypatch):
    module = configs('termination_functions', 'final')
    monkeypatch.setattr(module, 'time_out', lambda env: env.episode_length_buf >= env.max_episode_length)
    env = environment()
    env.max_episode_length = 500
    radius = NS(params={'position_tolerance': .02})
    env.reward_manager = NS(get_term_cfg=lambda _: radius)
    env.scene['object_collection'].data.object_pos_w[:, 0, 0] = torch.tensor([0., .02, .021, float('nan')])
    for step in (0, 499, 500, 501):
        env.episode_length_buf = torch.full((4,), step)
        success, timeout = module.success_at_timeout(env), module.unsuccessful_timeout(env)
        assert not (success & timeout).any()
        assert torch.equal(success | timeout, env.episode_length_buf >= 500)
        assert success.tolist() == ([True, True, False, False] if step >= 500 else [False] * 4)
    radius.params['position_tolerance'] = .01
    assert module.success_at_timeout(env).tolist() == [True, False, False, False]
    terms = configs('terminations', 'final').FinalV1TerminationsCfg()
    assert terms.success.time_out and terms.time_out.time_out


def test_shelf_termination_is_union(configs, monkeypatch):
    module = configs('termination_functions', 'final')
    a, b = torch.tensor([False, True, False, True]), torch.tensor([False, False, True, True])
    monkeypatch.setattr(module, 'shelf_collision', lambda *args: a)
    monkeypatch.setattr(module, 'shelf_displacement', lambda *args: b)
    assert torch.equal(module.shelf_termination(None), a | b)


def test_manager_names_schedules_and_legacy_isolation(configs):
    rewards = configs('rewards', 'final')
    cfg, old = rewards.FinalV1RewardsCfg(), rewards.FinalRewardsCfg()
    assert {name for name, term in vars(cfg).items() if hasattr(term, 'weight')} == {
        'hand_reaching', 'goal_distance', 'goal_hold', 'hand_contact', 'action_rate_penalty',
        'joint_velocity_penalty', 'object_penalty', 'object_speed_penalty', 'shelf_penalty',
        'obstacle_contact_penalty', 'wrench_change_penalty'}
    for name in ('action_rate_penalty', 'joint_velocity_penalty', 'object_penalty', 'shelf_penalty'):
        assert getattr(cfg, name).weight == -1.
    cfg.goal_hold.params['position_tolerance'] = 10
    assert old.goal_hold.params['position_tolerance'] == .02
    curriculum = configs('curricula', 'final').FinalV1CurriculumCfg()
    expected = {'object_free_acceleration': (50., 20.), 'object_acceleration_weight': (0., -.05),
                'wrench_free_force_change': (50., 20.), 'wrench_free_moment_change': (5., 2.),
                'goal_position_tolerance': (.02, .01)}
    schedule = runpy.run_path(str(MODULE_ROOT/'mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py'))['scheduled_value']
    for name, (initial, final) in expected.items():
        params = getattr(curriculum, name).params
        start = 4000 if name.startswith('goal') else 3000
        assert schedule(start, params['points'], params['interval']) == initial
        assert schedule(10000, params['points'], params['interval']) == final
        assert schedule(90000, params['points'], params['interval']) == final
        assert schedule(0, params['points'], params['interval']) == initial
        assert schedule((start + 10000)//2, params['points'], params['interval']) == pytest.approx((initial+final)/2)
    spawn = NS(params={})
    env = NS(cfg=NS(max_gray_obstacles=10, max_white_obstacles=10),
             event_manager=NS(get_term_cfg=lambda _:spawn, set_term_cfg=lambda *_: None))
    term = curriculum.obstacle_population
    for it, pair in [(0,(1,1)), (1999,(1,1)), (2000,(1,2)), (3000,(2,2)), (4000,(2,3)),
                     (14000,(7,8)), (15000,(8,8)), (90000,(8,8)), (0,(1,1))]:
        env.learning_iteration = it
        result = term.func(env, None, **term.params)
        assert (result['gray_obstacles'],result['white_obstacles']) == pair


def test_population_fallback_order_and_exhaustion(geometry, monkeypatch):
    assert list(geometry.population_attempts(2, 2)) == [(2,2),(2,1),(1,1),(1,0),(0,0)]
    assert list(geometry.population_attempts(0, 2)) == [(0,2),(0,1),(0,0)]
    calls = []
    def fail(self, target, goal):
        calls.append((self.config.gray_obstacles, self.config.obstacles-self.config.gray_obstacles))
        np.testing.assert_allclose(target, geometry.shelf_xy_to_generator_cm([[0,0]]))
        np.testing.assert_allclose(goal, geometry.shelf_xy_to_generator_cm([[0,.15]]))
        raise geometry.GenerationError(np.array([0]),np.array([0]),np.array([15]),{})
    monkeypatch.setattr(geometry.FinalLayoutGenerator, 'generate_batch', fail)
    _, _, config = geometry.sample_target_goal(np.random.default_rng(0), np.array([.04,.04]), False,.02,.15,.02,.5)
    centers, _, gray, white = geometry.sample_population(np.random.default_rng(0),np.array([0.,0.]),
        np.array([0.,.15]), np.full((21,2),.04),10,10,10,config,0,attempts_per_count=1)
    assert (gray,white)==(0,0) and centers.shape==(0,2)
    assert calls == list(geometry.population_attempts(10,10))[:-1]


@pytest.mark.parametrize('wide', [False,True])
def test_target_goals_fit_scaled_shapes_and_keep_nontrivial_motion(geometry,wide):
    rng=np.random.default_rng(32)
    targets=[]
    for _ in range(1000):
        half=rng.uniform(.025,.047,2)*rng.uniform(.9,1.1)
        target,goal,cfg=geometry.sample_target_goal(rng,half,wide,.02,.15,.02,math.pi/6)
        assert np.all(np.abs(target)+half+.02 <= [.20,.50]+np.zeros(2)+1e-10)
        assert np.all(np.abs(goal)+half+.02 <= [.20,.50]+np.zeros(2)+1e-10)
        assert .05-1e-10 <= np.linalg.norm(goal-target) <= .15+1e-10
        direction=(goal-target)/np.linalg.norm(goal-target)
        assert abs(direction[1]) >= .5-1e-10
        cfg.check()
        if not wide: assert np.linalg.norm(target)<=.02
        else: assert np.all(np.abs(target)<=[.16,.40])
        targets.append(target)
    if wide:
        assert np.ptp(targets,axis=0).min() > .15


@pytest.mark.parametrize('counts', [(1,1),(8,8),(10,10)])
def test_real_generator_returns_safe_layout_or_reduced_population(geometry,counts):
    rng=np.random.default_rng(42)
    for wide in (False,True):
        half=rng.uniform(.025,.045,(21,2))*1.1
        half[0]=np.array([.054,.035])*1.1  # Largest mug with full Gray padding.
        target,goal,cfg=geometry.sample_target_goal(rng,half[0],wide,.02,.15,.02,.5)
        centers,yaw,gray,white=geometry.sample_population(rng,target,goal,half,10,*counts,cfg,.15,attempts_per_count=2)
        indices=[0,*range(1,1+gray),*range(11,11+white)]
        assert geometry.valid_layout(np.concatenate((target[None],centers)),yaw[indices],half[indices])
        assert gray<=counts[0] and white<=counts[1]


@pytest.mark.parametrize('kind,parts', [('cube',1),('mug',20),('bottle',4)])
def test_new_shapes_have_closed_convex_pieces_and_flat_bottom(kind,parts):
    from collections import Counter
    mesh=runpy.run_path(str(MODULE_ROOT/'assets/contact_sweep_clutter_fusion_holding_osc_object_random_geometry.py'))['flat_collision_mesh']
    tree=ast.parse((MODULE_ROOT/'assets/final_spawners.py').read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='shelf_object_meshes']
    ns={'math':math,'flat_collision_mesh':mesh}
    exec(compile(tree,'mesh','exec'),ns)
    pieces=ns['shelf_object_meshes'](kind)
    assert len(pieces)==parts
    assert min(p[2] for points,_,_ in pieces for p in points)==0
    for points,counts,indices in pieces:
        offset=0;edges=Counter();volume=0.
        points=np.asarray(points)
        for count in counts:
            face=indices[offset:offset+count];offset+=count
            for a,b in zip(face,face[1:]+face[:1]):edges[a,b]+=1
            for j in range(1,count-1):volume+=np.dot(points[face[0]],np.cross(points[face[j]],points[face[j+1]]))/6
        assert all(edges[b,a]==n==1 for (a,b),n in edges.items())
        assert volume>0


def test_partial_scene_reset_preserves_other_rows_and_scaled_bottoms(load_mdp, monkeypatch):
    events = importlib.import_module('_baseline_cpu_pkg.mdp.event_functions.final')
    monkeypatch.setattr(events, 'apply_gravity_mask', lambda *args: None)
    monkeypatch.setattr(events, 'apply_final_visibility', lambda *args, **kwargs: None)
    n, capacity = 3, 21
    position = torch.zeros(n, capacity, 3)
    rotation = torch.zeros(n, 4);rotation[:,0]=1
    states = torch.full((n, capacity, 13), 77.)
    objects = NS(num_objects=capacity, data=NS(object_pos_w=position),
        write_object_link_state_to_sim=lambda value, env_ids: states.__setitem__(env_ids,value))
    bounds = np.tile(np.array([[-.04,-.04,-.002],[.04,.04,.10]]),(n,capacity,1,1))
    env = NS(num_envs=n,device='cpu',_population_layout_seed=7,_final_object_bounds=bounds,
        cfg=NS(max_gray_obstacles=10,max_white_obstacles=10,obstacle_observation_slots=20),
        scene={'object_collection':objects,'shelf':NS(data=NS(root_pos_w=torch.zeros(n,3),root_quat_w=rotation)),
               'robot':NS(data=NS(root_pos_w=torch.zeros(n,3),root_quat_w=rotation))},
        _active_object_mask=torch.zeros(n,capacity,dtype=torch.bool),target_id=torch.zeros(n,1,dtype=torch.long),
        target_width=torch.zeros(n,1),sweep_dir=torch.zeros(n,3),
        _initial_obstacle_observation=torch.full((n,80),77.))
    ids = torch.tensor([1])
    events.randomize_final_scene(env, ids, gray_obstacles=10, white_obstacles=10, wide_target=True)
    assert (states[[0,2]]==77).all()
    assert (env._initial_obstacle_observation[[0,2]]==77).all()
    active=env._active_object_mask[1]
    assert active[0]
    torch.testing.assert_close(states[1,active,2],torch.full_like(states[1,active,2],1.052))
    assert (states[1,~active,2]==100).all()
    assert torch.isfinite(states[1]).all()
    assert .05 <= env.sweep_dir[1].norm() <= .15+1e-6
    snapshot = states[1].clone()
    env._final_layout_rng=np.random.default_rng(7)
    events.randomize_final_scene(env, ids, gray_obstacles=10, white_obstacles=10, wide_target=True)
    torch.testing.assert_close(states[1],snapshot)


def test_shelf_merge_keeps_collision_only_terminal_cost(final_rewards, make_env):
    env=make_env(4)
    env.step_dt=.02
    shelf=env.scene['shelf'].data
    shelf.root_pos_w=shelf.default_root_state[:,:3]+env.scene.env_origins
    shelf.root_vel_w=torch.zeros(4,6)
    shelf.root_vel_w[1,0]=.2
    shelf.root_pos_w[2,0]+=.1  # Displacement-only termination must not add -3.
    env.scene['wrist_frame'].data.target_pos_w[:,0,2]=shelf.root_pos_w[:,2]+1.2
    env.scene['wrist_frame'].data.target_pos_w[3,0,2]=shelf.root_pos_w[3,2]+1.06
    env.scene['finger_frame']=NS(data=NS(target_pos_w=torch.zeros(4,2,3)))
    env.termination_manager=NS(get_term_cfg=lambda _:NS(params={}))
    old=final_rewards.ShelfCollisionPenalty(NS(params={}),env)
    merged=final_rewards.ShelfPenalty(NS(params={}),env)
    collision=torch.tensor([False,True,False,True])
    torch.testing.assert_close(-merged(env)*env.step_dt,-3*collision.float()-.5*old(env)*env.step_dt)


from test_clutter_v1_models import _algorithm, runner_configs
from test_clutter_fusion_models import fusion_models


def test_final_ppo_consumes_57_and_80_dimensions_and_updates(fusion_models, runner_configs, tmp_path):
    import sys
    from tensordict import TensorDict
    ns=sys.modules['isaaclab_rl.rsl_rl'].__dict__
    module=_execute_definitions(MODULE_ROOT/'agent/final_ppo_cfg.py',ns)
    obs=TensorDict({'policy':torch.randn(8,57),'privileged':torch.randn(8,3),
                   'obstacles':torch.randn(8,80)},batch_size=[8])
    obs['obstacles'][:,::4]=1
    spec=importlib.util.spec_from_file_location('_final_model_compat', PROJECT_ROOT/'IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py')
    compat=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    for cls in (module.FinalV1PPORunnerCfg,module.FinalV2PPORunnerCfg):
        cfg=compat.handle_deprecated_rsl_rl_cfg(cls(), '5.0.1')
        alg=_algorithm(obs,cfg)
        assert alg.actor.get_latent(obs).shape==(8,73)
        assert alg.critic.get_latent(obs).shape==(8,76)
        with torch.inference_mode():
            for _ in range(4):
                action=alg.act(obs)
                alg.process_env_step(obs,-action.square().sum(-1),torch.zeros(8,dtype=torch.long),{})
            alg.compute_returns(obs)
        assert all(np.isfinite(v) for v in alg.update().values())
        path=tmp_path/f'{cfg.experiment_name}.pt'
        torch.save(alg.save(),path)
        restored=_algorithm(obs,cfg)
        assert restored.load(torch.load(path,weights_only=True),None,strict=True)
        torch.testing.assert_close(alg.actor(obs),restored.actor(obs),rtol=0,atol=0)
        exported=torch.jit.script(alg.actor.as_jit())
        flat=torch.cat((obs['policy'],obs['obstacles']),dim=-1)
        assert flat.shape[-1]==137
        torch.testing.assert_close(exported(flat),alg.actor(obs))
