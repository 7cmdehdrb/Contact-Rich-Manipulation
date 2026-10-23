"""CPU checks of the physical EEF costs and real RewardManager rate budgets."""

import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions


@pytest.fixture
def safety(load_mdp, monkeypatch):
    from _source_units import import_unit
    return import_unit("contact_sweep_clutter_fusion.safety_rewards")


@pytest.fixture
def rate_env(safety, make_env):
    env = make_env(3)
    env.step_dt = 0.02
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.current_world_wrench = torch.zeros(3, 6)
    env.current_positive_rate = torch.ones(3)
    reader = SimpleNamespace(world_wrench=lambda env: env.current_world_wrench)
    env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(wrist_wrench=SimpleNamespace(func=reader))),
    )
    cfg = SimpleNamespace(weight=-0.1, params={})
    penalty = safety.WrenchRatePenalty(cfg, env)
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
    manager._term_names = ["positive", "existing_cost", "wrench_rate"]
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


def advance(env):
    env.common_step_counter += 1
    env._sim_step_counter += 2


def test_initial_contact_gentle_changes_and_constant_load_are_free(rate_env):
    env, penalty, manager = rate_env
    manager._episode_sums["positive"].fill_(100.0)
    env.current_world_wrench.fill_(1000.0)
    assert torch.equal(penalty(env), torch.zeros(3))
    advance(env)
    env.current_world_wrench[:, 0] += 40.0  # exactly 2000 N/s
    env.current_world_wrench[:, 3] += 2.0  # exactly 100 N m/s
    assert torch.equal(penalty(env), torch.zeros(3))
    advance(env)
    assert torch.equal(penalty(env), torch.zeros(3))


def test_separate_force_moment_units_and_combined_exponential_max(rate_env):
    env, penalty, manager = rate_env
    manager._episode_sums["positive"].fill_(100.0)
    penalty(env)
    advance(env)
    env.current_world_wrench[:] = torch.tensor([
        [70.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # force rate 3500: severity 0.5
        [0.0, 0.0, 0.0, 3.5, 0.0, 0.0],  # moment rate 175: severity 0.5
        [70.0, 0.0, 0.0, 3.5, 0.0, 0.0],
    ])
    expected = math.expm1(1.5) / math.expm1(3.0)
    assert torch.allclose(penalty(env), torch.full((3,), expected))
    previous = penalty._previous_wrench.clone()
    assert torch.allclose(penalty(env), torch.full((3,), expected))
    assert torch.equal(penalty._previous_wrench, previous)
    advance(env)
    assert torch.equal(penalty(env), torch.zeros(3))


def test_absolute_cap_nonfinite_guard_and_actual_time_derivative(rate_env):
    env, penalty, manager = rate_env
    manager._episode_sums["positive"].fill_(100.0)
    penalty(env)
    advance(env)
    env.current_world_wrench[:, 0] = torch.tensor([1.0e30, math.inf, math.nan])
    assert torch.equal(penalty(env), torch.ones(3))
    assert penalty._previous_valid.tolist() == [True, False, False]
    penalty.reset()
    env.current_world_wrench.zero_()
    advance(env)
    penalty(env)
    advance(env)
    env.step_dt = 0.01
    env.current_world_wrench[:, 0] = 70.0  # same delta now means 7000 N/s
    assert torch.equal(penalty(env), torch.ones(3))


def test_real_manager_never_exceeds_absolute_or_positive_return_budget(rate_env):
    env, _, manager = rate_env
    manager.compute(env.step_dt)
    for step in range(1, 31):
        advance(env)
        env.current_world_wrench[:, 0] = 200.0 * (step % 2)
        env.current_positive_rate[:] = torch.tensor([0.0, 0.01, 20.0])
        manager.compute(env.step_dt)
        assert torch.all(-manager._episode_sums["wrench_rate"] <= 0.05 * manager._episode_sums["positive"] + 1.0e-7)
        assert torch.all(manager._step_reward[:, -1].abs() <= 0.1 + 1.0e-7)
    assert manager._episode_sums["wrench_rate"].tolist() == pytest.approx([-0.001, -0.0013, -0.06])
    assert torch.all(manager._episode_sums["existing_cost"] < 0)


def test_no_earned_reward_means_no_penalty(rate_env):
    env, _, manager = rate_env
    env.current_positive_rate.zero_()
    manager.compute(env.step_dt)
    advance(env)
    env.current_world_wrench[:, 0] = 200.0
    manager.compute(env.step_dt)
    assert torch.equal(manager._episode_sums["wrench_rate"], torch.zeros(3))


def test_partial_reset_and_stale_sample_do_not_create_rates(rate_env):
    env, penalty, manager = rate_env
    manager._episode_sums["positive"].fill_(100.0)
    penalty(env)
    advance(env)
    env.current_world_wrench[:, 0] = 200.0
    manager.compute(env.step_dt)
    manager.reset(torch.tensor([1]))
    assert penalty._previous_valid.tolist() == [True, False, True]
    advance(env)
    env.current_world_wrench[:, 0] = 0.0
    env.sensor_data_fresh[2] = False
    manager.compute(env.step_dt)
    assert manager._step_reward[:, -1].tolist() == pytest.approx([-0.1, 0.0, 0.0])
    advance(env)
    env.sensor_data_fresh[:] = True
    env.current_world_wrench[:, 0] = 200.0
    assert penalty(env)[2] == 0.0  # first fresh sample restores history only
    advance(env)
    env.current_world_wrench.zero_()
    assert penalty(env)[2] == 1.0


def test_wrong_order_and_positive_weight_are_rejected(rate_env):
    env, penalty, manager = rate_env
    manager._term_names.reverse()
    manager._term_cfgs.reverse()
    with pytest.raises(RuntimeError, match="follow every positive"):
        penalty(env)
    manager._term_names.reverse()
    manager._term_cfgs.reverse()
    manager.get_term_cfg("wrench_rate").weight = 0.1
    with pytest.raises(ValueError, match="negative"):
        penalty(env)


@pytest.mark.parametrize("kwargs", [
    {"free_force_rate": -1}, {"free_moment_rate": math.nan},
    {"saturation_force_rate": 2000}, {"saturation_moment_rate": math.inf},
    {"sharpness": 0}, {"sharpness": 100}, {"max_positive_fraction": 0.051},
])
def test_invalid_rate_parameters_rejected(rate_env, kwargs):
    env, penalty, _ = rate_env
    with pytest.raises(ValueError):
        penalty(env, **kwargs)


def test_orientation_upward_only_boundary_shape_and_finite_cap(safety, make_env):
    env = make_env(7)
    angles = torch.tensor([0.0, math.acos(0.75), math.pi / 3, math.pi / 2, math.pi, 0.0, 0.0])
    quaternion = env.scene["ee_frame"].data.target_quat_w[:, 0]
    quaternion[:] = torch.stack((torch.cos(angles / 2), torch.sin(angles / 2), torch.zeros(7), torch.zeros(7)), dim=-1)
    quaternion[5] *= -1  # quaternion sign is physically irrelevant
    quaternion[6] = math.nan
    actual = safety.eef_upward_z_orientation_cost(env)
    assert torch.isfinite(actual).all()
    assert actual.tolist() == pytest.approx([1.0, 0.25, 0.0, 0.0, 0.0, 1.0, 1.0], abs=1e-6)
    with pytest.raises(ValueError):
        safety.eef_upward_z_orientation_cost(env, onset_angle=math.pi)


def test_eef_speed_uses_point_velocity_and_rotation_not_com_or_position_difference(safety, make_env):
    env = make_env(4)
    robot = env.scene["robot"]
    robot.body_names = ["wrist_3_link", "axia80_link"]
    robot.data.body_link_pos_w = torch.zeros(4, 2, 3)
    robot.data.body_link_pos_w[:, 1, 0] = 10.0
    robot.data.body_link_lin_vel_w = torch.zeros(4, 2, 3)
    robot.data.body_link_ang_vel_w = torch.zeros(4, 2, 3)
    env.scene["ee_frame"].data.target_pos_w[:, 0] = torch.tensor([10.0, 0.0, 0.5])
    robot.data.body_link_ang_vel_w[:, 1, 1] = torch.tensor([2.0, 4.0, 4.0, 0.0])
    # Pure rotation produces 1m/s and 2m/s; translation cancels the third TCP.
    robot.data.body_link_lin_vel_w[2, 1, 0] = -2.0
    robot.data.body_link_lin_vel_w[3, 1, 0] = math.inf
    robot.data.body_com_lin_vel_w = torch.full((4, 2, 3), 999.0)
    robot.data.body_lin_vel_w = torch.full((4, 2, 3), 999.0)
    expected = [0.0, 0.5 * math.expm1(1.0), 0.0, 100.0]
    assert safety.eef_linear_speed_cost(env).tolist() == pytest.approx(expected)
    assert safety.eef_linear_speed_cost(env).tolist() == pytest.approx(expected)
