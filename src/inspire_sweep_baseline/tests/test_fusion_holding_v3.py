"""Cumulative hold budget, absolute curricula and Gray-only Red intrusion."""

from dataclasses import replace
import importlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from conftest import MODULE_ROOT
from test_fusion_holding_v0 import environment, curricula
from test_fusion_holding_v2 import cfg
from test_omni_configuration import _execute_definitions, omni_config


@pytest.fixture
def capped(load_mdp):
    return importlib.import_module(
        "_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter_fusion_holding_scheduled"
    ).CappedGoalSettling


def test_budget_counts_paid_time_pauses_on_exit_and_resets_only_selected_rows(capped):
    env = environment()
    term = capped(NS(), env)
    pos = env.scene["object_collection"].data.object_pos_w
    pos[1, 0, 0] = .02  # Half-rate at the boundary still consumes full paid time.
    env.scene["object_collection"].data.object_lin_vel_w[2, 0, 0] = .05
    total = torch.zeros(4)
    for _ in range(75):
        total += term(env) * env.step_dt
    pos[0, 0, 0] = .03
    pos[3, 0, 2] = 1.2
    for _ in range(200):
        rate = term(env)
        total += rate * env.step_dt
        assert rate[0] == rate[3] == 0
    pos[0, 0, 0] = 0
    pos[3, 0, 2] = 1.05
    for _ in range(100):
        total += term(env) * env.step_dt
    assert total.tolist() == pytest.approx([3., 1.5, 1.5, 3.], abs=1e-5)
    assert term._paid_steps.tolist() == [150] * 4
    assert not term(env).any()
    term.reset([1, 3])
    assert term(env).tolist() == pytest.approx([0., .5, 0., 1.])
    term.reset()
    assert not term._paid_steps.any()


@pytest.mark.parametrize("dt", [.02, .03, .07, .4])
def test_exact_time_cap_and_no_reentry_payment(capped, dt):
    env = environment()
    env.step_dt = dt
    term = capped(NS(), env)
    integral = sum((term(env)[0].item() * dt) for _ in range(500))
    assert integral == pytest.approx(3., abs=1e-6)
    env.scene["object_collection"].data.object_pos_w[:, 0, 0] = .1
    assert not term(env).any()
    env.scene["object_collection"].data.object_pos_w[:, 0, 0] = 0
    assert not term(env).any()


@pytest.mark.parametrize("duration", [0., -1., float("nan"), float("inf")])
def test_invalid_duration_rejected(capped, duration):
    env = environment()
    with pytest.raises(ValueError):
        capped(NS(), env)(env, max_hold_time=duration)


def test_sustained_success_still_pays_all_150_steps(capped, load_mdp):
    module = importlib.import_module("_baseline_cpu_pkg.mdp.termination_functions.contact_sweep_clutter_fusion_holding")
    env = environment()
    goal = NS(params={"position_tolerance": .02, "speed_scale": .05})
    env.reward_manager = NS(get_term_cfg=lambda _: goal)
    reward, success = capped(NS(), env), module.ScheduledHoldingSuccess(NS(), env)
    for step in range(150):
        # Match ManagerBasedRLEnv's termination-before-reward order.
        assert success(env).all().item() == (step == 149)
        assert reward(env).tolist() == [1.] * 4
    assert not reward(env).any()


@pytest.fixture
def random_curriculum(cfg, omni_config):
    cc = omni_config.baseline.configclass
    schedule = importlib.import_module(
        "_baseline_cpu_pkg.mdp.curriculum_functions.contact_sweep_clutter_fusion_holding_osc_object_random"
    ).gray_zone_padding_schedule
    return _execute_definitions(
        MODULE_ROOT / "mdp/curricula/contact_sweep_clutter_fusion_holding_osc_object_random.py",
        dict(configclass=cc, FusionHoldingV0CurriculumCfg=type(cfg), FusionHoldingV2CurriculumCfg=type(cfg),
             CurriculumTermCfg=omni_config.terms.RewardTermCfg, gray_zone_padding_schedule=schedule),
    ).ObjectRandomV2CurriculumCfg()


def test_population_and_padding_boundaries_resume_rewind(random_curriculum, cfg):
    population = importlib.import_module("_baseline_cpu_pkg.mdp.curriculum_functions.contact_sweep_clutter_fusion").obstacle_population_schedule
    spawn = NS(params={})
    env = NS(cfg=NS(max_gray_obstacles=8, max_white_obstacles=8),
             event_manager=NS(get_term_cfg=lambda _: spawn, set_term_cfg=lambda *_: None))
    term = random_curriculum.obstacle_population
    assert term.params["interval"] == 2000  # Each color alternates every 1000.
    checks = [(0, 1, 1), (2999, 1, 1), (3000, 1, 2), (3999, 1, 2),
              (4000, 2, 2), (5000, 2, 3), (10999, 5, 5), (11000, 5, 6),
              (11999, 5, 6), (12000, 6, 6), (15000, 6, 6), (16000, 6, 6), (90000, 6, 6)]
    for it, gray, white in checks + list(reversed(checks)):
        env.learning_iteration = it
        assert population(env, None, **term.params) == dict(gray_obstacles=gray, white_obstacles=white)
    padding = random_curriculum.gray_zone_padding
    checks = [(0, 0), (5999, 0), (6000, 0), (6249, 0), (6250, .00625),
              (9000, .075), (11999, .14375), (12000, .15), (50000, .15)]
    for it, expected in checks + list(reversed(checks)):
        env.learning_iteration = it
        assert padding.func(env, None, **padding.params) == pytest.approx(expected)
        assert spawn.params["gray_red_padding_fraction"] == pytest.approx(expected)
    assert cfg.obstacle_population.params["interval"] == 3000
    assert random_curriculum.obstacle_population.params["final_gray"] == 6
    assert random_curriculum.obstacle_population.params["final_white"] == 6


def test_gray_expansion_admits_old_red_centers_but_white_stays_unchanged(load_mdp):
    module = importlib.import_module(
        "_baseline_cpu_pkg.mdp.common.contact_sweep_clutter_fusion_holding_osc_object_random_placement"
    )
    layout = importlib.import_module("_baseline_cpu_pkg.assets.scenes.contact_sweep_clutter_layout")
    base = layout.DEFAULT_LAYOUT_CONFIG
    generator = module.GrayPaddingLayoutGenerator(seed=42, config=replace(base, obstacles=8, gray_obstacles=4),
                                                 gray_red_padding_fraction=.15)
    target, direction, distance = np.array([[0., 20.]]), np.array([[1., 0.]]), np.array([15.])
    points = np.array([[[7.5, 29.0], [7.5, 28.7], [7.5, 30.4], [7.5, 34.1], [-9., 20.]]])
    old = layout.scene_generator.zone_masks(points, target, direction, distance, base)
    red, gray, rear = generator.zone_masks(points, target, direction, distance, "gray")
    assert old[0][0, 0] and gray[0, 0] and not red[0, 0]
    assert red[0, 1] and not gray[0, 1]
    assert gray[0, 2] and not gray[0, 3]
    assert old[0][0, 4] and rear[0, 4]
    for a, b in zip(generator.zone_masks(points, target, direction, distance, "white"), old):
        np.testing.assert_array_equal(a, b)
    # Execute production rejection sampling, independently classify every result.
    targets = np.repeat(target, 40, axis=0)
    goals = targets + [15., 0.]
    obstacles = generator.generate(targets, goals)
    directions = np.repeat(direction, 40, axis=0)
    distances = np.repeat(distance, 40)
    old_red, old_gray, _ = layout.scene_generator.zone_masks(obstacles, targets, directions, distances, base)
    new_red, new_gray, new_rear = generator.zone_masks(obstacles, targets, directions, distances, "gray")
    assert not new_red[:, :4].any() and new_gray[:, :4].all() and new_rear[:, 0].all()
    assert (old_red[:, :4]).any()  # Padding survives all other rejection filters.
    assert not (old_red[:, 4:] | old_gray[:, 4:]).any()
    assert np.all(obstacles >= base.object_bounds[0]) and np.all(obstacles <= base.object_bounds[1])
    all_centers = np.concatenate((targets[:, None], obstacles), axis=1)
    distances2 = ((all_centers[:, :, None] - all_centers[:, None, :]) ** 2).sum(-1)
    i, j = np.triu_indices(all_centers.shape[1], 1)
    assert (distances2[:, i, j] >= (2 * base.radius + base.gap_factor * base.finger_thickness) ** 2).all()
