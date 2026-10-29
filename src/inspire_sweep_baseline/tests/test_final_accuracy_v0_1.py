"""Accuracy v0_1 contracts using real configclasses and production tensor functions."""
import ast
import importlib
import math
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from conftest import MODULE_ROOT
from test_final_configuration import configs, assert_ordered_equal
from test_final_versions import final_rewards
from test_omni_configuration import omni_config, _execute_definitions
from test_clutter_v1_models import runner_configs

REWARD_NAMES = {
    'hand_reaching': 'target_reaching', 'goal_distance': 'target_goal_distance',
    'goal_hold': 'target_goal_hold', 'hand_contact': 'target_contact',
    'action_rate_penalty': 'robot_action_rate_penalty',
    'joint_velocity_penalty': 'robot_joint_velocity_penalty',
    'wrench_change_penalty': 'wrist_wrench_change_penalty',
}
CURRICULUM_NAMES = {
    'object_free_speed': 'target_free_speed', 'goal_deadzone': 'target_goal_direction_deadzone',
    'wrench_free_force_change': 'wrist_wrench_free_force_change',
    'wrench_free_moment_change': 'wrist_wrench_free_moment_change',
    'wrench_weight': 'wrist_wrench_change_weight',
    'obstacle_free_force': 'obstacle_contact_free_force', 'obstacle_weight': 'obstacle_contact_weight',
}
TERMINATION_NAMES = dict(zip(
    ('object_drop', 'object_speed', 'shelf_termination', 'joint_velocity', 'success', 'time_out'),
    ('object_fall_or_tilt', 'target_speed', 'shelf_failure', 'arm_joint_velocity',
     'target_goal_success_at_timeout', 'target_goal_miss_at_timeout')))


def active(cfg):
    return {k: v for k, v in vars(cfg).items() if hasattr(v, 'func')}


@pytest.mark.parametrize('role,suffix,mapping', [
    ('rewards', 'RewardsCfg', REWARD_NAMES),
    ('curricula', 'CurriculumCfg', CURRICULUM_NAMES),
    ('terminations', 'TerminationsCfg', TERMINATION_NAMES),
])
def test_order_names_and_parent_isolation(configs, role, suffix, mapping):
    module = configs(role, 'final')
    parent_cls = getattr(module, 'FinalGoalAccuracy' + suffix)
    child_cls = getattr(module, 'FinalGoalAccuracyV0_1' + suffix)
    parent = parent_cls()
    snapshot = parent.to_dict()
    child = child_cls()
    assert child_cls.__bases__ == (parent_cls,)
    assert list(active(child)) == [mapping.get(name, name) for name in active(parent)]
    assert len(active(child)) == {'rewards': 11, 'curricula': 11, 'terminations': 6}[role]
    for old, new in mapping.items():
        assert getattr(child, old) is None
        assert getattr(child, new) is not None
    for name, term in active(parent).items():
        new = active(child)[mapping.get(name, name)]
        expected = term.to_dict()
        if role == 'rewards' and name in ('hand_reaching', 'goal_distance', 'goal_hold', 'hand_contact'):
            continue  # Explicit functional changes are covered numerically below.
        if role == 'rewards' and name == 'shelf_penalty':
            expected['params'] = {'termination_term_name': 'shelf_failure'}
        if role == 'curricula' and expected['params'].get('term_name') == 'wrench_change_penalty':
            expected['params']['term_name'] = 'wrist_wrench_change_penalty'
        if role == 'terminations' and name in ('success', 'time_out'):
            expected['params'] = {'goal_hold_term_name': 'target_goal_hold'}
        assert new.to_dict() == expected
    for term in active(child).values():
        term.params['isolation_probe'] = True
    assert_ordered_equal(parent.to_dict(), snapshot)
    assert_ordered_equal(parent_cls().to_dict(), snapshot)
    assert all('isolation_probe' not in t.params for t in active(child_cls()).values())


def test_hold_contact_and_timeout_boundaries(configs, final_rewards, monkeypatch):
    rewards = configs('rewards', 'final').FinalGoalAccuracyV0_1RewardsCfg()
    terms = configs('terminations', 'final').FinalGoalAccuracyV0_1TerminationsCfg()
    distance = torch.tensor([0., .01, .02-1e-6, .02, .02+1e-6, .03, .04, .05, .06, .07])
    speed, valid = torch.zeros_like(distance), torch.ones_like(distance, dtype=torch.bool)
    monkeypatch.setattr(final_rewards, 'target_measurements', lambda *a: (distance, speed, valid))
    monkeypatch.setattr(final_rewards, 'target_hand_contact_force_w', lambda env: torch.zeros(len(distance), 3))
    endings = configs('termination_functions', 'final')
    monkeypatch.setattr(endings, '_object_position', lambda env: torch.stack((distance, speed, speed), -1))
    monkeypatch.setattr(endings, 'time_out', lambda env: env.expired)
    calls = []
    env = NS(sensor_data_fresh=torch.ones_like(valid),
             reward_manager=NS(get_term_cfg=lambda name: getattr(rewards, name)),
             command_manager=NS(get_command=lambda name: torch.zeros(len(distance), 3)),
             expired=torch.ones_like(valid), record_goal_accuracy_metrics=lambda: calls.append(1))
    hold = rewards.target_goal_hold
    values = hold.weight * hold.func(env, **hold.params)
    torch.testing.assert_close(values[[0, 1, 3, 5, 6, 7, 8, 9]],
        torch.tensor([16., 8.408964, 4., 2.25, 1., .25, 0., 0.]))
    assert abs(values[2]-values[4]) < .002
    contact = rewards.target_contact
    torch.testing.assert_close(contact.func(env, **contact.params), (distance <= .02).float())
    success, miss = terms.target_goal_success_at_timeout, terms.target_goal_miss_at_timeout
    assert success.time_out and miss.time_out
    torch.testing.assert_close(success.func(env, **success.params), distance <= .02)
    torch.testing.assert_close(miss.func(env, **miss.params), distance > .02)
    env.expired[:] = False
    assert not success.func(env, **success.params).any()
    assert not miss.func(env, **miss.params).any()
    speed[:] = .05
    torch.testing.assert_close(hold.weight*hold.func(env, **hold.params), values/2)
    valid[:] = False
    assert not hold.func(env, **hold.params).any()
    assert not contact.func(env, **contact.params).any()
    assert len(calls) == 2


def test_contact_selects_target_not_other_objects_or_shelf(configs, final_rewards, monkeypatch):
    n = 6
    matrices = [torch.zeros(n, 1, 2, 3) for _ in range(2)]
    matrices[0][0, 0, 0, 0] = 3  # non-target contact only
    matrices[1][1, 0, 0, 0] = .01  # target exactly at threshold
    matrices[0][2, 0, :, 0] = torch.tensor([2., -2.])  # vector cancellation
    matrices[0][3, 0, 0, 0] = 3  # stale target sample
    matrices[0][4, 0, 0, 0] = float('nan')
    matrices[1][5, 0, 0, 0] = float('nan')  # non-target NaN must not leak
    matrices[0][5, 0, 0, 0] = 1
    scene = {'object_collection': NS(object_names=['a', 'b']),
             'whole_hand_contact': NS(data=NS(net_forces_w=torch.full((n, 1, 3), 100.)))}
    for name, matrix in zip(['a', 'b'], matrices):
        scene['hand_contact_'+name] = NS(data=NS(force_matrix_w=matrix))
    cfg = configs('rewards', 'final').FinalGoalAccuracyV0_1RewardsCfg()
    env = NS(num_envs=n, device='cpu', target_id=torch.tensor([[1], [1], [0], [0], [0], [0]]),
             scene=scene, sensor_data_fresh=torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.bool),
             reward_manager=NS(get_term_cfg=lambda name: getattr(cfg, name)))
    d = torch.full((n,), .1); valid = torch.zeros(n, dtype=torch.bool)
    monkeypatch.setattr(final_rewards, 'target_measurements', lambda *a: (d, d, valid))
    term = cfg.target_contact
    assert term.func(env, **term.params).tolist() == [0, 1, 0, 0, 0, 1]
    d[:] = .02; valid[:] = True
    assert term.func(env, **term.params).tolist() == [1]*n  # OR, no freshness/speed gate for goal


@pytest.mark.parametrize('q,expected', [(.02, 13.1455976697), (0., 18.9036993765)])
def test_exponential_reward_actual_50hz_return(configs, q, expected):
    term = configs('rewards', 'final').FinalGoalAccuracyV0_1RewardsCfg().target_goal_distance
    t = torch.arange(1, 501, dtype=torch.float64)*.02
    d = .15-(.15-q)*((t-3)/3).clamp(0, 1)
    states = torch.zeros(500, 1, 13, dtype=torch.float64); states[:, 0, 0] = d
    env = NS(num_envs=500, device='cpu', target_id=torch.zeros(500, 1, dtype=torch.long),
             scene={'object_collection': NS(data=NS(object_link_state_w=states))},
             command_manager=NS(get_command=lambda name: torch.zeros(500, 3)))
    rewards = term.weight*term.func(env, **term.params)*.02
    assert rewards.sum().item() == pytest.approx(expected, abs=1e-9)
    assert (rewards[t<=3] > 0).all()
    torch.testing.assert_close(rewards, term.weight*term.func(env, **term.params)*.02)


def test_reaching_live_target_partial_reset_and_goal_independence(configs):
    term = configs('rewards', 'final').FinalGoalAccuracyV0_1RewardsCfg().target_reaching
    state = torch.zeros(2, 1, 13)
    eef = torch.zeros(2, 1, 3); eef[:, 0, 0] = .4
    env = NS(num_envs=2, device='cpu', step_dt=.02, target_id=torch.zeros(2, 1, dtype=torch.long),
             scene={'object_collection': NS(data=NS(object_link_state_w=state)),
                    'robot': NS(data=NS()), 'ee_frame': NS(data=NS(target_pos_w=eef))})
    fn = term.func(term, env)
    eef[:, 0, 0] = .3
    torch.testing.assert_close(fn(env, **term.params)*env.step_dt, torch.tensor([.1, .1]))
    state[:, 0, 0] = .1  # moving target, still no goal access
    torch.testing.assert_close(fn(env, **term.params)*env.step_dt, torch.tensor([.1, .1]))
    state[:, 0, 0] = 0
    fn.reset([0])
    eef[:, 0, 0] = .2
    torch.testing.assert_close(fn(env, **term.params)*env.step_dt, torch.tensor([.1, 0.]))


def test_schedules_renamed_apply_and_rewind(configs):
    rewards = configs('rewards', 'final').FinalGoalAccuracyV0_1RewardsCfg()
    curriculum = configs('curricula', 'final').FinalGoalAccuracyV0_1CurriculumCfg()
    old = configs('curricula', 'final').FinalGoalAccuracyCurriculumCfg()
    old_rewards = configs('rewards', 'final').FinalGoalAccuracyRewardsCfg()
    for term in (rewards.wrist_wrench_change_penalty, old_rewards.wrench_change_penalty):
        term.func = term.func(term, NS(num_envs=1, device='cpu'))
    def manager(cfg):
        return NS(get_term_cfg=lambda name: getattr(cfg, name),
                  set_term_cfg=lambda name, value: setattr(cfg, name, value))
    env, parent = NS(reward_manager=manager(rewards)), NS(reward_manager=manager(old_rewards))
    for iteration in (0, 499, 500, 1900, 3000, 6000, 10000, 12000, 500):
        env.learning_iteration = parent.learning_iteration = iteration
        for old_name, term in active(old).items():
            if 'term_name' not in term.params:
                continue
            new = getattr(curriculum, CURRICULUM_NAMES.get(old_name, old_name))
            assert term.func(parent, None, **term.params) == new.func(env, None, **new.params)
            a = getattr(old_rewards, term.params['term_name'])
            b = getattr(rewards, new.params['term_name'])
            assert a.params == b.params and a.weight == b.weight
    assert curriculum.target_goal_speed_scale is curriculum.target_goal_position_tolerance is None


def test_registry_ppo_and_env_inheritance(configs, runner_configs):
    registrations = {}
    gym = NS(registry=registrations, register=lambda **kw: registrations.update({kw['id']: kw}))
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym': gym})
    assert not any('Accuacy' in name for name in registrations)
    entry = registrations['Isaac-Inspire-Final-GoalAccuracy-v0_1']
    assert entry['entry_point'] == registrations['Isaac-Inspire-Final-GoalAccuracy-v0']['entry_point']
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':FinalGoalAccuracyV0_1EnvCfg')
    tree = ast.parse((MODULE_ROOT/'envs/final_envs.py').read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='FinalGoalAccuracyV0_1EnvCfg')
    assert [ast.unparse(b) for b in cfg.bases] == ['FinalGoalAccuracyEnvCfg']
    ppo = _execute_definitions(MODULE_ROOT/'agent/final_ppo_cfg.py', sys.modules['isaaclab_rl.rsl_rl'].__dict__)
    assert ppo.FinalGoalAccuracyV0_1PPORunnerCfg().to_dict() == {
        **ppo.FinalGoalAccuracyPPORunnerCfg().to_dict(), 'experiment_name': 'inspire_final_goal_accuracy_v0_1'}


@pytest.mark.parametrize('scale', [.9, 1., 1.1])
def test_cuboid_collision_bounds_and_legacy_cube(load_mdp, scale):
    geometry = importlib.import_module('_baseline_cpu_pkg.assets.contact_sweep_clutter_fusion_holding_osc_object_random_geometry')
    tree = ast.parse((MODULE_ROOT/'assets/final_spawners.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='shelf_object_meshes']
    ns = {'math': math, 'flat_collision_mesh': geometry.flat_collision_mesh}
    exec(compile(tree, 'production-mesh', 'exec'), ns)
    mesh = ns['shelf_object_meshes']
    def points(size=None):
        pieces = mesh('cube') if size is None else mesh('cube', cube_size=size)
        return np.array([p for vertices, _, _ in pieces for p in vertices])*scale
    old, new = points(), points((.06, .08, .10))
    np.testing.assert_allclose(old.max(0)-old.min(0), np.array([.08]*3)*scale)
    np.testing.assert_allclose(new.max(0)-new.min(0), np.array([.06, .08, .10])*scale)
    assert new[:, 2].min() == 0
    for yaw in [0, .3, math.pi/2]:
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        placed = new @ rotation.T + [0, 0, 1.05-new[:, 2].min()]
        assert placed[:, 2].min() == pytest.approx(1.05)


def test_all_terms_pass_official_argument_validation(configs, omni_config):
    import inspect
    from conftest import PROJECT_ROOT
    tree = ast.parse((PROJECT_ROOT/'IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ManagerBase')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_resolve_common_term_cfg')
    ns = dict(inspect=inspect, ManagerTermBaseCfg=omni_config.terms.ManagerTermBaseCfg,
              ManagerTermBase=importlib.import_module('isaaclab.managers').ManagerTermBase)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'official-validator', 'exec'), ns)
    for role, suffix, min_argc in [('rewards', 'RewardsCfg', 1), ('curricula', 'CurriculumCfg', 2),
                                  ('terminations', 'TerminationsCfg', 1)]:
        cfg = getattr(configs(role, 'final'), 'FinalGoalAccuracyV0_1'+suffix)()
        for name, term in active(cfg).items():
            ns[fn.name](NS(_env=NS(sim=NS(is_playing=lambda:False))), name, term, min_argc=min_argc)


def test_renamed_wrench_budget_preserves_order_and_cap(configs):
    totals = []
    for cls, name in [('FinalGoalAccuracyRewardsCfg', 'wrench_change_penalty'),
                      ('FinalGoalAccuracyV0_1RewardsCfg', 'wrist_wrench_change_penalty')]:
        cfg = getattr(configs('rewards', 'final'), cls)()
        term = getattr(cfg, name); term.weight = -.1
        env = NS(num_envs=2, device='cpu', step_dt=.02)
        term.func = term.func(term, env)
        names = list(active(cfg))
        sums = {k: torch.zeros(2) for k in names}
        # Same accumulated rewards under the name mapping, independent of shape.
        for k in names[:4]: sums[k][:] = torch.tensor([2., 4.])
        sums[name][:] = torch.tensor([-.1, -.9])
        env.reward_manager = NS(active_terms=names, get_term_cfg=lambda k: getattr(cfg, k), _episode_sums=sums)
        totals.append(term.func._reward_budget(env, .05))
    torch.testing.assert_close(totals[0], totals[1])
    torch.testing.assert_close(totals[1], torch.tensor([150., 0.]))


def test_v0_2_adds_only_timeout_bonus(configs, runner_configs):
    module = configs('rewards', 'final')
    parent = module.FinalGoalAccuracyV0_1RewardsCfg()
    snapshot = parent.to_dict()
    child = module.FinalGoalAccuracyV0_2RewardsCfg()
    expected_order = list(active(parent))
    expected_order.insert(expected_order.index('wrist_wrench_change_penalty'), 'target_goal_timeout_bonus')
    assert list(active(child)) == expected_order
    assert_ordered_equal(
        {k: v for k, v in child.to_dict().items() if k != 'target_goal_timeout_bonus'}, snapshot)
    child.target_goal_hold.params['position_tolerance'] = .5
    assert_ordered_equal(parent.to_dict(), snapshot)
    assert_ordered_equal(module.FinalGoalAccuracyV0_1RewardsCfg().to_dict(), snapshot)
    tree = ast.parse((MODULE_ROOT/'envs/final_envs.py').read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FinalGoalAccuracyV0_2EnvCfg')
    assert [ast.unparse(b) for b in cfg.bases] == ['FinalGoalAccuracyV0_1EnvCfg']
    assert [n.target.id for n in cfg.body if isinstance(n, ast.AnnAssign)] == ['rewards']
    registrations = {}
    gym = NS(registry=registrations, register=lambda **kw: registrations.update({kw['id']: kw}))
    _execute_definitions(MODULE_ROOT/'__init__.py', {'gym': gym})
    entry = registrations['Isaac-Inspire-Final-GoalAccuracy-v0_2']
    assert entry['entry_point'] == registrations['Isaac-Inspire-Final-GoalAccuracy-v0_1']['entry_point']
    assert entry['kwargs']['env_cfg_entry_point'].endswith(':FinalGoalAccuracyV0_2EnvCfg')
    assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith(':FinalGoalAccuracyV0_2PPORunnerCfg')
    ppo = _execute_definitions(MODULE_ROOT/'agent/final_ppo_cfg.py', sys.modules['isaaclab_rl.rsl_rl'].__dict__)
    assert ppo.FinalGoalAccuracyV0_2PPORunnerCfg().to_dict() == {
        **ppo.FinalGoalAccuracyV0_1PPORunnerCfg().to_dict(), 'experiment_name': 'inspire_final_goal_accuracy_v0_2'}


def test_v0_2_wrench_budget_includes_same_step_timeout_bonus(configs):
    cfg = configs('rewards', 'final').FinalGoalAccuracyV0_2RewardsCfg()
    term = cfg.wrist_wrench_change_penalty
    env = NS(num_envs=2, device='cpu', step_dt=.02)
    term.func = term.func(term, env)
    term.weight = -.1  # Exercise the guard after the curriculum enables it.
    names = list(active(cfg))
    sums = {name: torch.zeros(2) for name in names}
    sums['target_reaching'][:] = 4.
    sums['wrist_wrench_change_penalty'][:] = -.2  # Existing 5% budget exhausted.
    env.reward_manager = NS(active_terms=names, get_term_cfg=lambda name: getattr(cfg, name),
                            _episode_sums=sums)
    # RewardManager updates each term's sum before computing the following term.
    sums['target_goal_timeout_bonus'][:] = torch.tensor([30., 0.])
    budget = term.func._reward_budget(env, .05)
    torch.testing.assert_close(budget, torch.tensor([750., 0.]))
    assert 'target_goal_timeout_bonus' in term.func._positive_names
    sums['target_goal_timeout_bonus'].zero_()
    torch.testing.assert_close(term.func._reward_budget(env, .05), torch.zeros(2))


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_v0_2_timeout_bonus_strict_xy_once_per_episode(configs, final_rewards, monkeypatch, dt):
    term = configs('rewards', 'final').FinalGoalAccuracyV0_2RewardsCfg().target_goal_timeout_bonus
    positions = torch.tensor([
        [0., 0., 10.], [.02-1e-6, 0., 0.], [.02, 0., 0.],
        [.02+1e-6, 0., 0.], [.015, .015, 0.], [0., -.019, -10.],
    ])
    monkeypatch.setattr(final_rewards, '_object_position', lambda env: positions)
    env = NS(step_dt=dt, expired=torch.zeros(6, dtype=torch.bool),
             command_manager=NS(get_command=lambda name: torch.zeros(6, 3)))
    def get_term(name):
        assert name == 'target_goal_success_at_timeout'
        return env.expired & (torch.linalg.vector_norm(positions[:, :2], dim=-1) <= .02)
    env.termination_manager = NS(get_term=get_term)
    def reward():
        return term.func(env, **term.params) * term.weight * env.step_dt
    total = torch.zeros(6)
    for _ in range(10):  # Goal occupancy alone must never pay the event.
        total += reward()
    torch.testing.assert_close(total, torch.zeros(6))
    env.expired[:] = True
    total += reward()
    torch.testing.assert_close(total, torch.tensor([30., 30., 0., 0., 0., 30.]))
    env.expired[:] = False  # Automatic reset starts a new episode.
    torch.testing.assert_close(reward(), torch.zeros(6))
    env.expired[0] = True  # Per-environment timeouts stay independent.
    torch.testing.assert_close(reward(), torch.tensor([30., 0., 0., 0., 0., 0.]))
