"""Behavioral checks for compensated wrench changes and actual reward budgets."""

import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions


@pytest.fixture
def v2_mdp(load_mdp, monkeypatch):
    from _source_units import import_unit
    return import_unit("contact_sweep_clutter_v2.mdp")


@pytest.fixture
def reward_env(v2_mdp, make_env):
    """Use the checked-in RewardManager.compute, with physical sample fixtures."""
    env = make_env(3)
    env.step_dt = 0.02
    env.current_world_wrench = torch.zeros(3, 6)
    env.current_positive_rate = torch.ones(3)
    reader = SimpleNamespace(world_wrench=lambda env: env.current_world_wrench)
    env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(wrist_wrench=SimpleNamespace(func=reader))),
    )
    cfg = SimpleNamespace(weight=-0.1, params={})
    penalty = v2_mdp.WrenchChangePenalty(cfg, env)
    cfg.func = penalty

    class ManagerBase:
        pass

    manager_cls = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py",
        {"torch": torch, "ManagerBase": ManagerBase},
        selected_classes={"RewardManager"},
    ).RewardManager
    manager = manager_cls.__new__(manager_cls)
    manager._env = env
    manager._term_names = ["positive", "existing_cost", "wrench_change"]
    manager._term_cfgs = [
        SimpleNamespace(func=lambda env: env.current_positive_rate, weight=1.0, params={}),
        SimpleNamespace(func=lambda env: torch.ones(env.num_envs), weight=-2.0, params={}),
        cfg,
    ]
    manager._class_term_cfgs = [cfg]
    manager._episode_sums = {name: torch.zeros(3) for name in manager.active_terms}
    manager._reward_buf = torch.zeros(3)
    manager._step_reward = torch.zeros(3, 3)
    env.reward_manager = manager
    env.max_episode_length_s = 10.0
    return env, penalty, manager


def test_first_fresh_sample_has_no_impulse_and_gentle_changes_are_free(reward_env):
    env, penalty, manager = reward_env
    env.current_world_wrench.fill_(100.0)
    assert torch.equal(penalty(env), torch.zeros(3))
    env.current_world_wrench[:, 0] += 0.4
    env.current_world_wrench[:, 3] += 0.009
    manager._episode_sums["positive"].fill_(100.0)
    assert torch.equal(penalty(env), torch.zeros(3))
    # Constant contact force, however large, has no change penalty.
    assert torch.equal(penalty(env), torch.zeros(3))


def test_force_and_moment_have_separate_units_and_quadratic_cost(reward_env):
    env, penalty, manager = reward_env
    penalty(env)
    manager._episode_sums["positive"].fill_(100.0)
    env.current_world_wrench[:] = torch.tensor([
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0.5*(2.5/5)^2
        [0.0, 0.0, 0.0, 0.06, 0.0, 0.0],  # 0.5*(0.05/0.1)^2
        [3.0, 0.0, 0.0, 0.06, 0.0, 0.0],
    ])
    assert torch.allclose(penalty(env), torch.tensor([0.125, 0.125, 0.25]))


def test_cost_has_absolute_unit_cap_and_no_dt_derivative(reward_env):
    env, penalty, manager = reward_env
    penalty(env)
    manager._episode_sums["positive"].fill_(100.0)
    env.current_world_wrench[:, 2] = 1.0e30
    assert torch.equal(penalty(env), torch.ones(3))
    penalty.reset()
    env.current_world_wrench.zero_()
    penalty(env)
    env.step_dt = 0.01
    env.current_world_wrench[:, 1] = 3.0
    assert torch.allclose(penalty(env), torch.full((3,), 0.125))


def test_cost_budget_is_causal_actual_positive_and_never_includes_other_costs(reward_env):
    env, _, manager = reward_env
    manager.compute(env.step_dt)
    for step in range(1, 31):
        env.current_world_wrench[:, 0] = 100.0 * (step % 2)
        # One stalled env, one poor run, one well-performing run.
        env.current_positive_rate[:] = torch.tensor([0.0, 0.01, 20.0])
        manager.compute(env.step_dt)
        earned = manager._episode_sums["positive"]
        spent = -manager._episode_sums["wrench_change"]
        assert torch.all(spent <= 0.05 * earned + 1.0e-7)
        assert torch.all(manager._step_reward[:, -1].abs() <= 0.1 + 1.0e-7)
    # Large inherited costs do not create extra budget or consume its meaning.
    assert torch.all(manager._episode_sums["existing_cost"] < 0.0)
    assert manager._episode_sums["wrench_change"][0] == pytest.approx(-0.001)
    assert manager._episode_sums["wrench_change"][1] == pytest.approx(
        -0.05 * manager._episode_sums["positive"][1].item(),
    )
    assert manager._episode_sums["wrench_change"][2] == pytest.approx(-0.06)


def test_without_any_earned_positive_reward_cost_is_zero(reward_env):
    env, _, manager = reward_env
    env.current_positive_rate.zero_()
    manager.compute(env.step_dt)
    env.current_world_wrench[:, 0] = 100.0
    manager.compute(env.step_dt)
    assert torch.equal(manager._episode_sums["wrench_change"], torch.zeros(3))


def test_partial_reset_clears_only_selected_history_and_budget(reward_env):
    env, penalty, manager = reward_env
    manager._episode_sums["positive"].fill_(100.0)
    penalty(env)
    env.current_world_wrench[:, 0] = 100.0
    manager.compute(env.step_dt)
    manager.reset(torch.tensor([1]))
    assert manager._episode_sums["positive"].tolist() == pytest.approx([100.02, 0.0, 100.02])
    assert not penalty._previous_valid[1]
    assert penalty._previous_valid[[0, 2]].all()
    env.current_world_wrench[:, 0] = 0.0
    manager.compute(env.step_dt)
    assert manager._step_reward[1, -1] == 0.0
    assert torch.allclose(manager._step_reward[[0, 2], -1], torch.full((2,), -0.1))
    assert manager._episode_sums["wrench_change"][1] == 0.0


def test_invalid_reset_reading_does_not_become_history(reward_env):
    env, penalty, manager = reward_env
    manager._episode_sums["positive"].fill_(100.0)
    penalty(env)
    env.sensor_data_fresh[1] = False
    env.current_world_wrench[:, 0] = 100.0
    assert penalty(env).tolist() == [1.0, 0.0, 1.0]
    env.sensor_data_fresh[:] = True
    env.current_world_wrench[:, 0] = 0.0
    assert penalty(env).tolist() == [1.0, 0.0, 1.0]
    env.current_world_wrench[:, 0] = 100.0
    assert penalty(env).tolist() == [1.0, 1.0, 1.0]


def test_penalty_rejects_wrong_term_order_and_weight(reward_env):
    env, penalty, manager = reward_env
    manager._term_names[:] = ["wrench_change", "existing_cost", "positive"]
    manager._term_cfgs.reverse()
    with pytest.raises(RuntimeError, match="follow every positive"):
        penalty(env)
    manager._term_names.reverse()
    manager._term_cfgs.reverse()
    penalty._term_name = None
    manager.get_term_cfg("wrench_change").weight = 0.1
    with pytest.raises(ValueError, match="negative"):
        penalty(env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"free_force_change": -1.0},
        {"free_moment_change": math.nan},
        {"force_change_scale": 0.0},
        {"moment_change_scale": math.inf},
        {"max_positive_fraction": 0.0501},
        {"max_positive_fraction": -0.01},
    ],
)
def test_penalty_rejects_invalid_parameters(reward_env, kwargs):
    env, penalty, _ = reward_env
    with pytest.raises(ValueError):
        penalty(env, **kwargs)
