"""Schedule boundaries, standalone managers, registration and parent isolation."""

import ast
import importlib
import math
from types import SimpleNamespace

import pytest

from conftest import MODULE_ROOT, PACKAGE_ROOT
from test_omni_configuration import _execute_definitions, omni_config


ROOT = MODULE_ROOT / "contact_sweep_clutter_fusion_v1_0"


@pytest.fixture
def schedules(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    delayed = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_fusion_v3.curriculum")
    goal = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_fusion.curriculum")
    population = importlib.import_module("inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.population")
    namespace = dict(configclass=lambda cls: cls, CurriculumTermCfg=SimpleNamespace,
                     delayed_reward_schedule=delayed.delayed_reward_schedule,
                     goal_deadzone_schedule=goal.goal_deadzone_schedule,
                     obstacle_population_schedule=population.obstacle_population_schedule)
    cfg = _execute_definitions(ROOT / "curriculum_cfg.py", namespace).ContactSweepClutterFusionV1_0CurriculumCfg()
    terms = {k: v for k, v in vars(type(cfg)).items() if not k.startswith("_")}
    rewards = {term.params["term_name"]: SimpleNamespace(params={}, weight=-1.0)
               for name, term in terms.items() if "term_name" in term.params}
    event = SimpleNamespace(params={"excluded_x_half_angle": math.pi / 6})
    env = SimpleNamespace(
        learning_iteration=0,
        cfg=SimpleNamespace(max_gray_obstacles=3, max_white_obstacles=3),
        reward_manager=SimpleNamespace(get_term_cfg=rewards.__getitem__, set_term_cfg=rewards.__setitem__),
        event_manager=SimpleNamespace(get_term_cfg=lambda _: event, set_term_cfg=lambda *_: None),
    )
    return env, terms, rewards, event


@pytest.mark.parametrize("name,start,first,interval,last,before_last,final", [
    ("goal_hold_tolerance", .05, .04, 500, 2500, .02, .01),
    ("wrench_free_force_change", 50., 49., 250, 10750, 11., 10.),
    ("wrench_free_moment_change", 5., 4.9, 250, 10750, 1.1, 1.),
    ("object_free_acceleration", .3, .29, 250, 7000, .06, .05),
])
def test_new_schedule_boundaries_resume_and_rewind(schedules, name, start, first, interval, last, before_last, final):
    env, terms, rewards, _ = schedules
    term = terms[name]
    for iteration, expected in ((0, start), (999, start), (1000, first),
                                (1000 + interval - 1, first), (last - 1, before_last),
                                (last, final), (90000, final), (0, start)):
        env.learning_iteration = iteration
        for ids in (None, [0], [1, 3]):
            assert term.func(env, ids, **term.params) == pytest.approx(expected)
            assert rewards[term.params["term_name"]].params[term.params["parameter"]] == pytest.approx(expected)


def test_deadzone_and_all_schedules_coexist_without_weight_mutation(schedules):
    env, terms, rewards, event = schedules
    assert len(terms) == 9
    for iteration, expected in ((0, 30), (3000, 30), (3499, 30), (3500, 25),
                                (5999, 5), (6000, 0), (90000, 0), (0, 30)):
        env.learning_iteration = iteration
        for term in terms.values():
            term.func(env, None, **term.params)
        assert math.degrees(event.params["excluded_x_half_angle"]) == pytest.approx(expected)
        assert all(term.weight == -1 for term in rewards.values())


def test_managers_have_no_inherited_terms_and_no_tactile_reward_or_weight_schedule():
    for filename in ("rewards_cfg.py", "terminations_cfg.py", "curriculum_cfg.py"):
        tree = ast.parse((ROOT / filename).read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert len(classes) == 1 and not classes[0].bases
    rewards = (ROOT / "rewards_cfg.py").read_text()
    assert "contact_count" not in rewards and "func=contact_any" not in rewards
    assert "action_rate_weight" not in (ROOT / "curriculum_cfg.py").read_text()


def test_four_fixed_local_penalties_and_contact_calibration(omni_config, load_mdp):
    cfgclass = omni_config.baseline.configclass
    goal = load_mdp("goal_rewards")
    placeholder = lambda *args, **kwargs: None
    namespace = dict(
        configclass=cfgclass, RewTerm=SimpleNamespace, SceneEntityCfg=SimpleNamespace,
        math=math, HAND_JOINT_NAMES=["hand"], arm_entity=lambda: SimpleNamespace(name="robot"),
        goal_rewards=goal,
        dex_rewards=SimpleNamespace(DexReachProgress=placeholder, DexPlanarGoalProgress=placeholder),
        combined_active_object_cost=placeholder, eef_linear_speed_cost=placeholder,
        goal_hold_xy=placeholder, shelf_failure=placeholder, ShelfCollisionPenalty=placeholder,
        ObjectAccelerationPenalty=placeholder, WrenchMagnitudeChangePenalty=placeholder,
        hand_contact_any=placeholder, obstacle_contact_force_cost=placeholder,
    )
    # SceneEntityCfg takes the positional entity name in production.
    namespace["SceneEntityCfg"] = lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    cfg = _execute_definitions(ROOT / "rewards_cfg.py", namespace).ContactSweepClutterFusionV1_0RewardsCfg()
    for name, func, weight in (
        ("arm_action_rate", goal.action_rate_l2, -.03),
        ("hand_action_rate", goal.action_rate_l2, -.03),
        ("arm_joint_velocity", goal.joint_vel_l2, -.03),
        ("hand_joint_velocity", goal.joint_vel_l2, -.01),
    ):
        term = getattr(cfg, name)
        assert term.func is func and term.weight == weight
    assert cfg.arm_action_rate.params["action_group"] == "arm"
    assert cfg.hand_action_rate.params["action_group"] == "hand"
    five_seconds = cfg.hand_contact.weight * 5
    assert five_seconds == pytest.approx(1.45)
    assert five_seconds / (12.990838965 + five_seconds) == pytest.approx(.1, abs=.001)


def test_env_direct_v3_inheritance_manager_replacement_and_parent_isolation(omni_config):
    configclass = omni_config.baseline.configclass

    @configclass
    class Parent:
        scene = SimpleNamespace(marker="parent")
        rewards = SimpleNamespace(eef_orientation=object())
        terminations = SimpleNamespace(push_fast=SimpleNamespace(params={"speed_condition": .8}))
        curriculum = SimpleNamespace(marker="parent")
        events = SimpleNamespace(object_spawn=SimpleNamespace(params={"excluded_x_half_angle": math.pi / 3}))

        def __post_init__(self):
            self.rewards.eef_orientation = None
            self.terminations.push_fast.params["speed_condition"] = .8

    @configclass
    class NewRewards:
        eef_orientation = None

    @configclass
    class NewTerminations:
        push_fast = SimpleNamespace(params={"speed_condition": .8})

    class Scene(SimpleNamespace):
        def configure_population_capacity(self, gray, white):
            self.capacity = (gray, white)

    namespace = dict(
        configclass=configclass, math=math, ContactSweepClutterFusionV3EnvCfg=Parent,
        ContactSweepClutterFusionV1_0RewardsCfg=NewRewards,
        ContactSweepClutterFusionV1_0TerminationsCfg=NewTerminations,
        ContactSweepClutterFusionV1_0CurriculumCfg=lambda: SimpleNamespace(marker="child"),
        ContactSweepClutterFusionV1_0SceneCfg=Scene,
        ContactSweepClutterFusionV1_0EventsCfg=lambda: SimpleNamespace(
            object_spawn=SimpleNamespace(params={"excluded_x_half_angle": math.pi / 6})),
    )
    cls = _execute_definitions(ROOT / "env_cfg.py", namespace).ContactSweepClutterFusionV1_0EnvCfg
    child = cls()
    assert cls.__bases__ == (Parent,)
    assert isinstance(child.rewards, NewRewards)
    assert isinstance(child.terminations, NewTerminations)
    assert child.curriculum.marker == "child"
    assert child.events.object_spawn.params["excluded_x_half_angle"] == pytest.approx(math.pi / 6)
    assert Parent().events.object_spawn.params["excluded_x_half_angle"] == pytest.approx(math.pi / 3)
    child.terminations.push_fast.params["speed_condition"] = 10
    assert cls().terminations.push_fast.params["speed_condition"] == .8

    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    spec = registrations["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0"]
    assert spec["entry_point"].endswith("fusion_v1_0.env:ContactSweepClutterFusionV1_0Env")
    assert spec["kwargs"]["env_cfg_entry_point"].endswith("fusion_v1_0.env_cfg:ContactSweepClutterFusionV1_0EnvCfg")
    assert spec["kwargs"]["rsl_rl_cfg_entry_point"].endswith("fusion_v1_0.ppo_cfg:ContactSweepClutterFusionV1_0PPORunnerCfg")
