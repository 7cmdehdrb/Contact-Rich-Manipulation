"""Offline verification of continuous schedules, boundaries and variant scope."""

import ast
import math
import runpy
from types import SimpleNamespace

import pytest

from conftest import MODULE_ROOT
from test_clutter_fusion_v1_variants import config_terms, params

linear = runpy.run_path(str(MODULE_ROOT/'contact_sweep_clutter_fusion_v1_3/curriculum.py'))['piecewise_linear_reward_threshold']


@pytest.mark.parametrize('name,points', [
    ('obstacle_free_force', [(1000,200.), (5000,100.), (10000,50.)]),
    ('wrench_free_force_change', [(1000,1000.), (5000,500.), (10000,100.), (20000,5.)]),
    ('wrench_free_moment_change', [(1000,100.), (5000,50.), (10000,10.), (20000,.5)]),
    ('object_free_acceleration', [(2000,100.), (5000,20.), (10000,10.), (20000,5.)]),
])
def test_actual_cfg_linear_boundaries_midpoints_resume_and_rewind(name, points):
    p = params(config_terms('1_3','curriculum_cfg.py')[name])['params']
    reward = SimpleNamespace(**params(config_terms('1_3','rewards_cfg.py')[p['term_name']]))
    assert reward.params[p['parameter']] == points[0][1]
    weight = reward.weight
    env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda _:reward, set_term_cfg=lambda *_:None))
    checks = [(0,points[0][1]), (points[0][0]-1,points[0][1]), *points, (90000,points[-1][1])]
    for (a,x),(b,y) in zip(points,points[1:]):
        checks += [(a+1,x+(y-x)/(b-a)), ((a+b)//2,(x+y)/2), (b-1,y-(y-x)/(b-a))]
    # Reversed order models restoring older checkpoints, not just forward training.
    for i,expected in checks + list(reversed(checks)):
        env.learning_iteration=i
        for ids in (None,[0],[1,3]):
            assert linear(env,ids,**p) == pytest.approx(expected)
            assert reward.params[p['parameter']] == pytest.approx(expected)
            assert reward.weight == weight


def test_only_requested_thresholds_and_schedules_change():
    parent=config_terms('1_1','rewards_cfg.py'); current=config_terms('1_3','rewards_cfg.py')
    assert parent.keys()==current.keys()
    for name in parent:
        if name not in ('obstacle_contact','wrench_change'):
            assert ast.dump(parent[name])==ast.dump(current[name])
    parent=config_terms('1_1','curriculum_cfg.py');current=config_terms('1_3','curriculum_cfg.py')
    assert parent.keys()==current.keys()
    for name in ('obstacle_population','pushing_free_speed','arm_free_speed','eef_free_speed','goal_deadzone'):
        assert ast.dump(parent[name])==ast.dump(current[name])
    assert 'goal_hold_tolerance' not in current


@pytest.mark.parametrize('points', [[], [(-1,1.)], [(1000,1.),(1000,0.)],
    [(1000,1.),(999,0.)], [(1.5,1.)], [(0,math.nan)], [(0,math.inf)], [(0,-1.)]])
def test_invalid_points_fail_before_manager_mutation(points):
    with pytest.raises(ValueError):
        linear(SimpleNamespace(learning_iteration=0),None,'term','parameter',points)
