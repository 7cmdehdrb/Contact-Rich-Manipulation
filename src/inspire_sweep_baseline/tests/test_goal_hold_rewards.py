"""Nonterminal goal achievement, sustained reward, and per-episode reset."""

from types import SimpleNamespace
import math

import pytest
import torch

from test_goal_terminations import _goal_env, _set_tilt


def _achievement_env(load_mdp, make_env, count=2, step_dt=0.02):
    module = load_mdp("goal_hold_rewards")
    rewards = load_mdp("goal_rewards")
    env, goal = _goal_env(make_env, count)
    env.step_dt = step_dt
    cfg = SimpleNamespace(params={"hold_time": 0.5, "failure_terms": ("push_fast",)})
    cfg.func = module.StableGoalAchievement(cfg, env)
    failed = torch.zeros(count, dtype=torch.bool)
    # There is deliberately no success termination or success termination cfg.
    env.termination_manager = SimpleNamespace(get_term=lambda name: failed if name == "push_fast" else None)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: cfg)
    return env, goal, cfg, rewards, failed


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.05])
def test_achievement_pays_once_and_hold_reward_continues_for_full_ten_seconds(load_mdp, make_env, step_dt):
    env, _, cfg, rewards, _ = _achievement_env(load_mdp, make_env, step_dt=step_dt)
    bonus, hold = torch.zeros(2), torch.zeros(2)
    required_steps = round(0.5 / step_dt)
    total_steps = round(10 / step_dt)
    for step in range(1, total_steps + 1):
        event = cfg.func(env, **cfg.params)
        rate = rewards.stable_goal_hold(env)
        assert event.count_nonzero().item() == (2 if step == required_steps else 0)
        assert torch.allclose(rate, torch.full((2,), min(step / required_steps, 1.0)))
        # Reading the hold reward twice must not advance time or re-pay a bonus.
        assert torch.equal(rate, rewards.stable_goal_hold(env))
        bonus += event * 100.0 * step_dt
        hold += rate * 20.0 * step_dt
    assert bonus.tolist() == pytest.approx([100.0, 100.0])
    expected_hold = 20 * step_dt * ((required_steps + 1) / 2 + total_steps - required_steps)
    assert hold.tolist() == pytest.approx([expected_hold, expected_hold], rel=1e-5)
    assert rewards.stable_goal_hold(env).tolist() == [1., 1.]


def test_departure_clears_hold_but_reentry_cannot_farm_bonus_and_partial_reset_rearms_it(load_mdp, make_env):
    env, _, cfg, rewards, _ = _achievement_env(load_mdp, make_env)
    for _ in range(24):
        assert cfg.func(env, **cfg.params).count_nonzero() == 0
    assert (cfg.func(env, **cfg.params) * env.step_dt).tolist() == [1., 1.]
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    position[0, 0] = 0.02
    assert cfg.func(env, **cfg.params).count_nonzero() == 0
    assert rewards.stable_goal_hold(env).tolist() == [0., 1.]
    position[0, 0] = 0.
    for step in range(1, 31):
        assert cfg.func(env, **cfg.params).count_nonzero() == 0
        assert rewards.stable_goal_hold(env)[0].item() == pytest.approx(min(step / 25, 1.))
    cfg.func.reset(torch.tensor([0]))
    assert rewards.stable_goal_hold(env).tolist() == [0., 1.]
    for _ in range(24):
        assert cfg.func(env, **cfg.params).count_nonzero() == 0
    assert (cfg.func(env, **cfg.params) * env.step_dt).tolist() == [1., 0.]
    cfg.func.reset()
    assert rewards.stable_goal_hold(env).count_nonzero() == 0
    for _ in range(24):
        assert cfg.func(env, **cfg.params).count_nonzero() == 0
    assert (cfg.func(env, **cfg.params) * env.step_dt).tolist() == [1., 1.]


@pytest.mark.parametrize("cause", ["position", "speed", "upright", "support", "footprint", "failure"])
def test_hold_and_bonus_share_all_stable_goal_gates(load_mdp, make_env, tensor_math, cause):
    env, goal, cfg, rewards, failed = _achievement_env(load_mdp, make_env, 1)
    for _ in range(24):
        assert cfg.func(env, **cfg.params).item() == 0.
    assert rewards.stable_goal_hold(env).item() == pytest.approx(0.96)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    if cause == "position":
        position[0, 0] = 0.011
    elif cause == "speed":
        env.scene["object_collection"].data.object_lin_vel_w[0, 0, 0] = 0.021
    elif cause == "upright":
        _set_tilt(env, tensor_math, math.radians(11))
    elif cause == "support":
        position[0, 2] += 0.02
    elif cause == "footprint":
        position[0, 0] = goal[0, 0] = 0.161
    else:
        failed[:] = True
    assert cfg.func(env, **cfg.params).item() == 0.
    assert rewards.stable_goal_hold(env).item() == 0.
