"""Offline CFG wiring and real schedule execution; no simulator imports."""

import ast
from copy import deepcopy
import math
import runpy
from types import SimpleNamespace

import pytest

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions


schedule = runpy.run_path(str(MODULE_ROOT / 'contact_sweep_clutter_fusion_v1_1/curriculum.py'))['staged_reward_threshold']


def config_terms(version, filename):
    path = MODULE_ROOT / f'contact_sweep_clutter_fusion_v{version}' / filename
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    assert not node.bases  # Reward/curriculum managers are independent declarations.
    return {n.targets[0].id: n.value for n in node.body if isinstance(n, ast.Assign)}


def params(call):
    return {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg in ('params', 'weight')}


@pytest.mark.parametrize('version', ['1_1', '1_2'])
def test_threshold_schedule_boundaries_resume_and_rewind(version):
    terms = config_terms(version, 'curriculum_cfg.py')
    rewards = config_terms(version, 'rewards_cfg.py')
    assert len(terms) == (9 if version == '1_1' else 10)
    expected = {
        'obstacle_free_force': (100., 50., 10000, 37),
        'object_free_acceleration': (100., 5., 20000, 77),
        'wrench_free_force_change': (500., 5., 20000, 77),
        'wrench_free_moment_change': (50., .5, 20000, 77),
    }
    for name, (initial, final, end, count) in expected.items():
        p = params(terms[name])['params']
        reward = SimpleNamespace(**deepcopy(params(rewards[p['term_name']])))
        assert reward.params[p['parameter']] == initial
        original_weight = reward.weight
        env = SimpleNamespace(reward_manager=SimpleNamespace(
            get_term_cfg=lambda _: reward, set_term_cfg=lambda *_: None))
        checks = [(0, initial), (999, initial), (1000, initial+(final-initial)/count),
                  (1249, initial+(final-initial)/count), (1250, initial+2*(final-initial)/count),
                  (end-1, final-(final-initial)/count), (end, final), (90000, final),
                  (2500, initial+7*(final-initial)/count), (0, initial)]
        for iteration, value in checks:
            env.learning_iteration = iteration
            for ids in (None, [0], [2, 3]):
                assert schedule(env, ids, **p) == pytest.approx(value)
                assert reward.params[p['parameter']] == pytest.approx(value)
                assert reward.weight == original_weight


def test_goal_two_stages_and_no_goal_schedule_in_v1_1():
    assert 'goal_hold_tolerance' not in config_terms('1_1', 'curriculum_cfg.py')
    p = params(config_terms('1_2', 'curriculum_cfg.py')['goal_hold_tolerance'])['params']
    reward = SimpleNamespace(params={'position_tolerance': .02})
    env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda _:reward, set_term_cfg=lambda *_:None))
    for iteration, expected in [(0,.02), (999,.02), (1000,.02-.01/17),
                                (4999,.01+.01/17), (5000,.01), (5249,.01),
                                (5250,.00975), (9999,.00525), (10000,.005),
                                (90000,.005), (5000,.01), (0,.02)]:
        env.learning_iteration = iteration
        assert schedule(env, None, **p) == pytest.approx(expected)
        assert reward.params['position_tolerance'] == pytest.approx(expected)


def test_reward_changes_are_limited_to_requested_parameters():
    previous = config_terms('1_0', 'rewards_cfg.py')
    first = config_terms('1_1', 'rewards_cfg.py')
    second = config_terms('1_2', 'rewards_cfg.py')
    assert previous.keys() == first.keys() == second.keys()
    for name in previous:
        if name not in ('obstacle_contact', 'wrench_change'):
            assert ast.dump(previous[name]) == ast.dump(first[name])
        if name not in ('goal_hold', 'hand_contact'):
            assert ast.dump(first[name]) == ast.dump(second[name])
    for version, goal, fade in [('1_1', .01, 0.), ('1_2', .02, .01)]:
        r = config_terms(version, 'rewards_cfg.py')
        assert params(r['goal_hold'])['params']['position_tolerance'] == goal
        assert params(r['hand_contact'])['params']['goal_contact_fade_width'] == fade
        assert params(r['hand_contact'])['weight'] == .29


@pytest.mark.parametrize('version,parent', [('1_1','1_0'), ('1_2','1_1'), ('1_1_1','1_1'), ('1_1_2','1_1'), ('1_3','1_1'), ('1_3_1','1_3')])
def test_registration_env_and_ppo_wiring(version, parent):
    registry = {}
    def register(id, **kwargs): registry[id] = kwargs
    ns = {'gym': SimpleNamespace(registry=registry, register=register)}
    _execute_definitions(MODULE_ROOT / '__init__.py', ns)
    task = registry[f'Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v{version}']
    assert task['entry_point'].endswith('fusion_v1_0.env:ContactSweepClutterFusionV1_0Env')
    assert task['kwargs']['env_cfg_entry_point'].endswith(f'fusion_v{version}.env_cfg:ContactSweepClutterFusionV{version}EnvCfg')
    assert task['kwargs']['rsl_rl_cfg_entry_point'].endswith(f'fusion_v{version}.ppo_cfg:ContactSweepClutterFusionV{version}PPORunnerCfg')
    for filename, suffix in [('env_cfg.py','EnvCfg'), ('ppo_cfg.py','PPORunnerCfg')]:
        tree = ast.parse((MODULE_ROOT/f'contact_sweep_clutter_fusion_v{version}'/filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        assert cls.bases[0].id == f'ContactSweepClutterFusionV{parent}{suffix}'
        if filename == 'ppo_cfg.py':
            assert len(cls.body) == 1
            assert cls.body[0].targets[0].id == 'experiment_name'
        else:
            assert {n.target.id for n in cls.body} == {'rewards', 'curriculum'}


@pytest.mark.parametrize('stages,interval', [([],250), ([(1000,2001,5.)],250),
    ([(1000,2000,5.),(2000,3000,1.)],250), ([(1000,2000,101.)],250),
    ([(1000,2000,math.nan)],250), ([(1000,2000,5.)],0)])
def test_invalid_schedule_rejected(stages, interval):
    with pytest.raises(ValueError):
        schedule(SimpleNamespace(learning_iteration=0), None, 'term', 'parameter', 100., stages, interval)


def test_v1_1_1_only_restores_force_allowances_and_removes_their_schedules():
    parent = config_terms('1_1', 'rewards_cfg.py')
    variant = config_terms('1_1_1', 'rewards_cfg.py')
    assert parent.keys() == variant.keys()
    for name in parent:
        if name not in ('obstacle_contact', 'wrench_change'):
            assert ast.dump(parent[name]) == ast.dump(variant[name])
    assert params(variant['obstacle_contact'])['params']['free_force'] == 2000.
    wrench = params(variant['wrench_change'])['params']
    assert wrench['free_force_change'] == 5000.
    assert wrench['free_moment_change'] == 500.
    schedules = config_terms('1_1_1', 'curriculum_cfg.py')
    original = config_terms('1_1', 'curriculum_cfg.py')
    assert schedules.keys() == original.keys() - {
        'obstacle_free_force', 'wrench_free_force_change', 'wrench_free_moment_change'}
    assert len(schedules) == 6
    for name in schedules:
        assert ast.dump(schedules[name]) == ast.dump(original[name])
    # All schedule targets are accounted for; none can overwrite fixed forces.
    targets = [params(call)['params'].get('term_name') for call in schedules.values()]
    assert 'obstacle_contact' not in targets and 'wrench_change' not in targets
    p = params(schedules['object_free_acceleration'])['params']
    reward = SimpleNamespace(params=deepcopy(params(variant['object_acceleration'])['params']))
    env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda _:reward, set_term_cfg=lambda *_:None))
    for iteration, value in [(0,100.), (999,100.), (1000,100.-95./77),
                             (19999,5.+95./77), (20000,5.), (90000,5.), (0,100.)]:
        env.learning_iteration = iteration
        assert schedule(env, None, **p) == pytest.approx(value)


def test_v1_1_2_changes_only_acceleration_allowance_and_its_schedule():
    parent = config_terms('1_1', 'rewards_cfg.py')
    variant = config_terms('1_1_2', 'rewards_cfg.py')
    assert parent.keys() == variant.keys()
    for name in parent:
        if name != 'object_acceleration':
            assert ast.dump(parent[name]) == ast.dump(variant[name])
    old = params(parent['object_acceleration'])
    new = params(variant['object_acceleration'])
    assert new['params'].pop('free_acceleration') == 1_000_000.
    old['params'].pop('free_acceleration')
    assert old == new
    assert ast.dump(parent['object_acceleration'].keywords[0]) == ast.dump(variant['object_acceleration'].keywords[0])
    schedules = config_terms('1_1_2', 'curriculum_cfg.py')
    original = config_terms('1_1', 'curriculum_cfg.py')
    assert schedules.keys() == original.keys() - {'object_free_acceleration'}
    assert len(schedules) == 8
    for name in schedules:
        assert ast.dump(schedules[name]) == ast.dump(original[name])
    assert all(params(call)['params'].get('term_name') != 'object_acceleration'
               for call in schedules.values())
