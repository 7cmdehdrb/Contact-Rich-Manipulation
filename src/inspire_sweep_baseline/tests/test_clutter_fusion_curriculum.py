"""Absolute schedule boundaries, rewind, partial resets, and goal immutability."""

import importlib
import math
from types import SimpleNamespace

import pytest

from conftest import MODULE_ROOT, PACKAGE_ROOT
from test_omni_configuration import _execute_definitions


@pytest.fixture
def schedule_env(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    reward = importlib.import_module("inspire_sweep_baseline.mdp.curriculum_functions.contact_sweep_clutter")
    goal = importlib.import_module("inspire_sweep_baseline.mdp.curriculum_functions.contact_sweep_clutter_fusion")
    cfg = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion/curriculum_cfg.py",
        {"configclass": lambda cls: cls, "CurriculumTermCfg": SimpleNamespace,
         "reward_schedule": reward.reward_schedule, "goal_deadzone_schedule": goal.goal_deadzone_schedule},
    ).ContactSweepClutterFusionCurriculumCfg()
    terms = [getattr(cfg, name) for name in vars(type(cfg)) if not name.startswith("_")]
    configs = {name: SimpleNamespace(weight=-1., params={"free_speed": 99.}) for name in
               ("pushing_speed", "arm_speed_excess", "action_rate", "eef_linear_speed")}
    configs["joint_velocity"] = SimpleNamespace(weight=-.03, params={"hand_ratio": 1 / 3})
    event = {"object_spawn": SimpleNamespace(params={"excluded_x_half_angle": 1.})}
    env = SimpleNamespace(
        learning_iteration=0,
        reward_manager=SimpleNamespace(get_term_cfg=configs.__getitem__, set_term_cfg=configs.__setitem__),
        event_manager=SimpleNamespace(get_term_cfg=event.__getitem__, set_term_cfg=event.__setitem__),
        current_goal=object(),
    )
    return env, terms, configs, event


@pytest.mark.parametrize("iteration,expected", [
    (0, (.1, 1., -.03, 1.5, 60)),
    (299, (.1, 1., -.03, 1.5, 60)),
    (300, (.09, .98, -.035, 1.48, 60)),
    (499, (.09, .98, -.035, 1.48, 60)),
    (500, (.09, .98, -.035, 1.48, 60)),
    (599, (.09, .98, -.035, 1.48, 60)),
    (600, (.08, .96, -.04, 1.46, 60)),
    (699, (.08, .96, -.04, 1.46, 60)),
    (700, (.08, .96, -.04, 1.46, 59)),
    (1000, (.07, .94, -.045, 1.44, 58)),
    (1500, (.05, .9, -.055, 1.4, 55)),
    (1800, (.05, .88, -.06, 1.38, 54)),
    (3000, (.05, .8, -.06, 1.3, 48)),
    (11999, (.05, .8, -.06, .72, 3)),
    (12000, (.05, .8, -.06, .7, 3)),
    (12499, (.05, .8, -.06, .7, 1)),
    (12500, (.05, .8, -.06, .7, 0)),
    (90000, (.05, .8, -.06, .7, 0)),
])
def test_schedules_apply_once_and_only_modify_future_goal_sampling(schedule_env, iteration, expected):
    env, terms, configs, event = schedule_env
    env.learning_iteration = iteration
    goal = env.current_goal
    for ids in (None, [0], [1, 3], None):
        for term in terms:
            term.func(env, ids, **term.params)
        actual = (
            configs["pushing_speed"].params["free_speed"],
            configs["arm_speed_excess"].params["free_speed"],
            configs["action_rate"].weight,
            configs["eef_linear_speed"].params["free_speed"],
            math.degrees(event["object_spawn"].params["excluded_x_half_angle"]),
        )
        assert actual == pytest.approx(expected)
        assert env.current_goal is goal
        assert configs["joint_velocity"].weight == -.03
        assert configs["joint_velocity"].params == {"hand_ratio": 1 / 3}
    env.learning_iteration = 0
    for term in terms:
        term.func(env, None, **term.params)
    assert configs["eef_linear_speed"].params["free_speed"] == 1.5
    assert event["object_spawn"].params["excluded_x_half_angle"] == math.pi / 3
