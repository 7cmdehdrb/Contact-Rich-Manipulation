"""Delayed Fusion schedules, endpoint clamps, rewind, and config isolation."""

import importlib
import math
from types import SimpleNamespace

import pytest

from conftest import MODULE_ROOT, PACKAGE_ROOT
from test_omni_configuration import _execute_definitions, omni_config


@pytest.fixture
def v1_schedule(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    delayed = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_fusion_v1.curriculum")
    goal = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_fusion.curriculum")
    namespace = {"configclass": lambda cls: cls, "CurriculumTermCfg": SimpleNamespace,
                 "goal_deadzone_schedule": goal.goal_deadzone_schedule,
                 "reward_schedule": object(), "delayed_reward_schedule": delayed.delayed_reward_schedule}
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/curriculum_cfg.py", namespace)
    cfg = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_v1/curriculum_cfg.py", namespace,
    ).ContactSweepClutterFusionV1CurriculumCfg()
    terms = {name: term for name, term in vars(type(cfg)).items() if not name.startswith("_")}
    configs = {name: SimpleNamespace(weight=-1., params={}) for name in
               ("pushing_speed", "arm_speed_excess", "action_rate", "eef_linear_speed",
                "contact_any", "contact_count", "wrench_change")}
    configs["joint_velocity"] = SimpleNamespace(weight=-.03, params={"hand_ratio": 1 / 3})
    event = SimpleNamespace(params={"excluded_x_half_angle": math.pi / 3})
    env = SimpleNamespace(
        learning_iteration=0,
        reward_manager=SimpleNamespace(get_term_cfg=configs.__getitem__, set_term_cfg=configs.__setitem__),
        event_manager=SimpleNamespace(get_term_cfg=lambda _: event, set_term_cfg=lambda *_: None),
        current_goal=object(),
    )
    return env, terms, configs, event


@pytest.mark.parametrize("name,initial,first,second,interval,last_iteration,penultimate,final", [
    ("pushing_free_speed", .1, .09, .08, 100, 1100, .04, .03),
    ("arm_free_speed", 1., .98, .96, 100, 1400, .82, .8),
    ("action_rate_weight", -.03, -.035, -.04, 100, 1000, -.055, -.06),
    ("eef_free_speed", 1.5, 1.48, 1.46, 500, 25000, .52, .5),
    ("contact_any_weight", .4, .42, .44, 100, 8400, 1.98, 2.),
    ("contact_count_weight", .4, .42, .44, 100, 8400, 1.98, 2.),
    ("wrench_free_force_rate", 2000., 1950., 1900., 100, 3400, 550., 500.),
    ("wrench_free_moment_rate", 100., 97.5, 95., 100, 3400, 27.5, 25.),
    ("wrench_saturation_force_rate", 5000., 4875., 4750., 100, 3400, 1375., 1250.),
    ("wrench_saturation_moment_rate", 250., 243.75, 237.5, 100, 3400, 68.75, 62.5),
])
def test_delayed_boundaries_endpoints_and_rewind(
    v1_schedule, name, initial, first, second, interval, last_iteration, penultimate, final,
):
    env, terms, configs, _ = v1_schedule
    term = terms[name]
    for iteration, expected in (
        (0, initial), (499, initial), (500, first), (500 + interval - 1, first),
        (500 + interval, second), (last_iteration - 1, penultimate),
        (last_iteration, final), (90000, final), (0, initial),
    ):
        env.learning_iteration = iteration
        for ids in (None, [0], [1, 3], None):
            assert term.func(env, ids, **term.params) == pytest.approx(expected)
            reward = configs[term.params["term_name"]]
            actual = reward.params[term.params["parameter"]] if "parameter" in term.params else reward.weight
            assert actual == pytest.approx(expected)


def test_all_schedules_coexist_without_changing_live_goal_or_joint_velocity(v1_schedule):
    env, terms, configs, event = v1_schedule
    goal = env.current_goal
    for iteration, degrees in ((0, 60), (1000, 60), (1099, 60), (1100, 59),
                               (6999, 1), (7000, 0), (90000, 0), (0, 60)):
        env.learning_iteration = iteration
        for ids in (None, [1], None):
            for term in terms.values():
                term.func(env, ids, **term.params)
            assert math.degrees(event.params["excluded_x_half_angle"]) == pytest.approx(degrees)
            assert env.current_goal is goal
            assert configs["contact_any"].weight == configs["contact_count"].weight
            wrench = configs["wrench_change"].params
            assert wrench["saturation_force_rate"] > wrench["free_force_rate"]
            assert wrench["saturation_moment_rate"] > wrench["free_moment_rate"]
            assert configs["joint_velocity"].weight == -.03
            assert configs["joint_velocity"].params == {"hand_ratio": 1 / 3}


def test_v1_registration_and_termination_override_isolate_parent(omni_config):
    configclass = omni_config.baseline.configclass

    @configclass
    class Parent:
        terminations = SimpleNamespace(push_fast=SimpleNamespace(params={"speed_condition": .8}))

        def __post_init__(self):
            self.parent_initialized = True

    @configclass
    class Curriculum:
        pass

    namespace = {"configclass": configclass, "ContactSweepClutterFusionEnvCfg": Parent,
                 "ContactSweepClutterFusionV1CurriculumCfg": Curriculum}
    child = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_fusion_v1/env_cfg.py", namespace,
    ).ContactSweepClutterFusionV1EnvCfg()
    assert child.parent_initialized
    assert child.terminations.push_fast.params["speed_condition"] == 1.5
    assert Parent().terminations.push_fast.params["speed_condition"] == .8
    child.terminations.push_fast.params["speed_condition"] = 9.
    assert type(child)().terminations.push_fast.params["speed_condition"] == 1.5
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    spec = registrations["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1"]
    assert spec["entry_point"] == registrations["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v0"]["entry_point"]
    assert spec["kwargs"]["env_cfg_entry_point"].endswith("fusion_v1.env_cfg:ContactSweepClutterFusionV1EnvCfg")
    assert spec["kwargs"]["rsl_rl_cfg_entry_point"].endswith("fusion_v1.ppo_cfg:ContactSweepClutterFusionV1PPORunnerCfg")


def test_real_configclass_copies_inherited_curriculum_terms(omni_config, v1_schedule):
    env, terms, _, _ = v1_schedule
    namespace = {
        "configclass": omni_config.baseline.configclass, "CurriculumTermCfg": SimpleNamespace,
        "reward_schedule": object(), "goal_deadzone_schedule": terms["goal_deadzone"].func,
        "delayed_reward_schedule": terms["pushing_free_speed"].func,
    }
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion/curriculum_cfg.py", namespace)
    _execute_definitions(MODULE_ROOT / "contact_sweep_clutter_fusion_v1/curriculum_cfg.py", namespace)
    parent = namespace["ContactSweepClutterFusionCurriculumCfg"]()
    child = namespace["ContactSweepClutterFusionV1CurriculumCfg"]()
    assert len(child.to_dict()) == 11
    child.goal_deadzone.params["interval"] = 9
    child.pushing_free_speed.params["final"] = .02
    assert parent.goal_deadzone.params["interval"] == 200
    assert parent.pushing_free_speed.params["minimum"] == .05
    fresh = type(child)()
    assert fresh.goal_deadzone.params["interval"] == 100
    assert fresh.pushing_free_speed.params["final"] == .03
