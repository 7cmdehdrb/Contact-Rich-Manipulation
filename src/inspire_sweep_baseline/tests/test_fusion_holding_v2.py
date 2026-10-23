"""Shared radius boundaries, contact OR, success timer and inherited schedules."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from conftest import MODULE_ROOT
from test_fusion_holding_v0 import curricula, environment
from test_omni_configuration import _execute_definitions, omni_config

ROOT = MODULE_ROOT / "contact_sweep_clutter_fusion_holding_v2"


@pytest.fixture
def mdp(load_mdp):
    load_mdp("goal_rewards")
    return importlib.import_module("_baseline_cpu_pkg.contact_sweep_clutter_fusion_holding_v2.mdp")


@pytest.fixture
def cfg(curricula, omni_config):
    return _execute_definitions(ROOT / "curriculum_cfg.py", dict(
        configclass=omni_config.baseline.configclass,
        CurriculumTermCfg=omni_config.terms.RewardTermCfg,
        FusionHoldingV0CurriculumCfg=type(curricula),
        reward_schedule=curricula.goal_position_scale.func,
    )).FusionHoldingV2CurriculumCfg()


def goal_env():
    env = environment()
    goal = NS(params={"position_tolerance": .02, "speed_scale": .05}, weight=4.)
    env.reward_manager = NS(get_term_cfg=lambda _: goal, set_term_cfg=lambda *_: None)
    env.scene["whole_hand_contact"] = NS(data=NS(net_forces_w=torch.zeros(4, 2, 3)))
    env.sensor_data_fresh = torch.ones(4, dtype=torch.bool)
    return env, goal


def update_radius(cfg, env, iteration):
    env.learning_iteration = iteration
    term = cfg.goal_position_tolerance
    return term.func(env, None, **term.params)


def test_radius_schedule_boundaries_resume_rewind_and_parent_isolation(cfg, curricula):
    env, goal = goal_env()
    checks = [(0,.02), (4999,.02), (5000,.02), (5001,.019998), (7500,.015),
              (9999,.010002), (10000,.01), (10001,.0099995), (15000,.0075),
              (19999,.0050005), (20000,.005), (90000,.005)]
    for iteration, expected in checks + list(reversed(checks)):
        assert update_radius(cfg, env, iteration) == pytest.approx(expected)
        assert goal.params["position_tolerance"] == pytest.approx(expected)
        assert goal.weight == 4.
    assert cfg.goal_position_scale is None
    assert curricula.goal_position_scale is not None
    for name, term in vars(curricula).items():
        if name != "goal_position_scale":
            assert getattr(cfg, name).to_dict() == term.to_dict()


@pytest.mark.parametrize("iteration", [0, 5000, 7500, 10000, 15000, 20000, 90000])
def test_all_three_consumers_share_inclusive_xy_radius(mdp, cfg, iteration):
    env, goal = goal_env()
    radius = update_radius(cfg, env, iteration)
    pos = env.scene["object_collection"].data.object_pos_w
    pos[:,0,0] = torch.tensor([0., radius, radius + 1e-5, radius * .8])
    pos[3,0,1] = radius * .8  # XY disk, not independent X/Y limits.
    assert (mdp.goal_settling(env, **goal.params) > 0).tolist() == [True, True, False, False]
    assert mdp.hand_contact_or_goal(env).tolist() == [1., 1., 0., 0.]
    success = mdp.ScheduledHoldingSuccess(NS(), env)
    for _ in range(149):
        assert not success(env).any()
    assert success(env).tolist() == [True, True, False, False]


def test_contact_force_threshold_freshness_and_no_double_count(mdp):
    env, goal = goal_env()
    env.scene["object_collection"].data.object_pos_w[:,0,0] = .03
    force = env.scene["whole_hand_contact"].data.net_forces_w
    force[:,0,0] = torch.tensor([.0099, .01, .01, float("nan")])
    env.sensor_data_fresh[2] = False
    assert mdp.hand_contact_or_goal(env).tolist() == [0., 1., 0., 0.]
    env.scene["object_collection"].data.object_pos_w[:,0,0] = 0.
    # Goal credit is independent of sensor freshness and does not double contact.
    assert mdp.hand_contact_or_goal(env).tolist() == [1., 1., 1., 1.]
    env.scene["object_collection"].data.object_pos_w[:,0,2] = 1.2
    assert mdp.hand_contact_or_goal(env).tolist() == [0., 1., 0., 0.]


def test_success_speed_failure_partial_reset_and_tightening(mdp, cfg):
    env, goal = goal_env()
    pos = env.scene["object_collection"].data.object_pos_w
    vel = env.scene["object_collection"].data.object_lin_vel_w
    pos[0,0,0] = .015
    vel[1,0,2] = .011  # Vertical velocity also prevents success, not contact credit.
    vel[2,0,0] = .01
    env.fail[3] = True
    success = mdp.ScheduledHoldingSuccess(NS(), env)
    for _ in range(150):
        result = success(env, failure_terms=("push_fast",))
    assert result.tolist() == [True, False, True, False]
    assert mdp.hand_contact_or_goal(env).tolist() == [1.] * 4
    success.reset([2])
    update_radius(cfg, env, 10000)
    assert not success(env, failure_terms=("push_fast",)).any()
    assert mdp.hand_contact_or_goal(env)[0] == 0
    assert mdp.goal_settling(env, **goal.params)[0] == 0
    update_radius(cfg, env, 0)
    assert not success(env, failure_terms=("push_fast",))[0]
    success.reset()
    assert not success._steps.any()


def test_settling_weight_and_time_scaling(mdp):
    env, goal = goal_env()
    pos = env.scene["object_collection"].data.object_pos_w
    vel = env.scene["object_collection"].data.object_lin_vel_w
    pos[1,0,0] = .02
    vel[2,0,0] = .05
    pos[3,0,0] = .021
    rates = mdp.goal_settling(env, **goal.params) * goal.weight
    assert rates.tolist() == pytest.approx([4., 2., 2., 0.])
    assert (rates * env.step_dt * 150).tolist() == pytest.approx([12., 6., 6., 0.])


def test_reward_termination_configs_and_registration(mdp, omni_config):
    cc = omni_config.baseline.configclass
    terms = omni_config.terms

    @cc
    class Rewards:
        goal_hold = terms.RewardTermCfg(func=lambda: None, weight=2.,
            params={"position_scale": .01, "speed_scale": .05, "cutoff_multiple": 3.})
        hand_contact = terms.RewardTermCfg(func=lambda: None, weight=1.45,
            params={"threshold_n": .01, "goal_contact_fade_width": 0.})

    @cc
    class Terminations:
        success = terms.TerminationTermCfg(func=lambda: None, params={
            "position_tolerance": .005, "speed_tolerance": .01, "hold_time": 3.,
            "failure_terms": ("push_fast",)})

    namespace = dict(configclass=cc, FusionHoldingV0RewardsCfg=Rewards,
                     goal_settling=mdp.goal_settling, hand_contact_or_goal=mdp.hand_contact_or_goal,
                     FusionHoldingV0TerminationsCfg=Terminations,
                     ScheduledHoldingSuccess=mdp.ScheduledHoldingSuccess)
    reward = _execute_definitions(ROOT / "rewards_cfg.py", namespace).FusionHoldingV2RewardsCfg()
    done = _execute_definitions(ROOT / "terminations_cfg.py", namespace).FusionHoldingV2TerminationsCfg()
    assert reward.goal_hold.weight == 4.
    assert reward.hand_contact.weight == 1.45
    assert reward.hand_contact.params == {"threshold_n": .01}
    assert reward.goal_hold.params == {"position_tolerance": .02, "speed_scale": .05}
    assert done.success.params == {"speed_tolerance": .01, "hold_time": 3., "failure_terms": ("push_fast",)}
    assert done.success.func is mdp.ScheduledHoldingSuccess
    assert Rewards().goal_hold.weight == 2.
    assert "goal_contact_fade_width" in Rewards().hand_contact.params
    registry = {}
    _execute_definitions(MODULE_ROOT / "__init__.py", dict(gym=NS(
        registry=registry, register=lambda **kw: registry.update({kw["id"]: kw}))))
    spec = registry["Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v2"]
    assert spec["kwargs"]["env_cfg_entry_point"].endswith("holding_v2.env_cfg:FusionHoldingV2EnvCfg")
    assert spec["kwargs"]["rsl_rl_cfg_entry_point"].endswith("holding_v2.ppo_cfg:FusionHoldingV2PPORunnerCfg")
