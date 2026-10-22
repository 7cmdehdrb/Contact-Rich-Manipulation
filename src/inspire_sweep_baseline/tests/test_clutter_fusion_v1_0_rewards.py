"""Physical reward contracts, reset isolation and real RewardManager budgets."""

import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions


@pytest.fixture
def rewards(load_mdp, monkeypatch):
    load_mdp("goal_rewards")
    loaded = {}
    for directory, names in (
        ("contact_sweep_clutter_fusion", ("safety_rewards",)),
        ("contact_sweep_clutter_fusion_v1_0", ("object_state", "contacts", "reward_functions")),
    ):
        package_name = f"_baseline_cpu_pkg.{directory}"
        package = ModuleType(package_name)
        package.__path__ = [str(MODULE_ROOT / directory)]
        monkeypatch.setitem(sys.modules, package_name, package)
        for name in names:
            fullname = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(fullname, MODULE_ROOT / directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, fullname, module)
            spec.loader.exec_module(module)
            loaded[name] = module
    return loaded["reward_functions"]


def advance(env):
    env.common_step_counter += 1
    env._sim_step_counter += 2


@pytest.fixture
def wrench_env(rewards, make_env):
    env = make_env(3)
    env.step_dt = 0.02
    env.common_step_counter = env._sim_step_counter = 0
    env.eef_wrench = torch.zeros(3, 6)
    env.current_positive_rate = torch.ones(3)

    class Reader:
        def __call__(self, env):
            return env.eef_wrench

        def world_wrench(self, env):
            raise AssertionError("Reward must call the exact EEF observation reader")

    env.observation_manager = SimpleNamespace(cfg=SimpleNamespace(policy=SimpleNamespace(
        wrist_wrench=SimpleNamespace(func=Reader(), params={}),
    )))
    cfg = SimpleNamespace(weight=-0.1, params={})
    penalty = rewards.WrenchMagnitudeChangePenalty(cfg, env)
    cfg.func = penalty
    manager_cls = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py",
        {"torch": torch, "ManagerBase": object}, selected_classes={"RewardManager"},
    ).RewardManager
    manager = manager_cls.__new__(manager_cls)
    manager._env = env
    manager._term_names = ["positive", "wrench_change"]
    manager._term_cfgs = [
        SimpleNamespace(func=lambda env: env.current_positive_rate, weight=1.0, params={}), cfg,
    ]
    manager._class_term_cfgs = [cfg]
    manager._episode_sums = {name: torch.zeros(3) for name in manager.active_terms}
    manager._episode_sums["positive"].fill_(100.0)
    manager._reward_buf = torch.zeros(3)
    manager._step_reward = torch.zeros(3, 2)
    env.reward_manager = manager
    env.max_episode_length_s = 10.0
    return env, penalty, manager


def test_wrench_exact_eef_sample_norm_before_delta_and_force_moment_thresholds(wrench_env):
    env, penalty, _ = wrench_env
    assert torch.equal(penalty(env), torch.zeros(3))
    advance(env)
    env.eef_wrench[:] = torch.tensor([
        [30., 40., 0., 3., 4., 0.],  # Net Force 50 N and Net Moment 5 Nm.
        [50., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 5., 0.],
    ])
    assert torch.equal(penalty(env), torch.zeros(3))
    advance(env)
    env.eef_wrench *= -1.0  # Direction reversal without magnitude change is free.
    assert torch.equal(penalty(env), torch.zeros(3))
    advance(env)
    env.eef_wrench[:, :3] = env.eef_wrench[:, :3].roll(1, dims=-1)
    env.eef_wrench[:, 3:] = env.eef_wrench[:, 3:].roll(1, dims=-1)
    assert torch.equal(penalty(env), torch.zeros(3))


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.1])
def test_wrench_delta_has_no_time_divisor_and_keeps_normalized_max_exponential(wrench_env, step_dt):
    env, penalty, _ = wrench_env
    env.step_dt = step_dt
    penalty(env)
    advance(env)
    env.eef_wrench[:] = torch.tensor([
        [87.5, 0., 0., 0., 0., 0.],
        [0., 0., 0., 8.75, 0., 0.],
        [87.5, 0., 0., 8.75, 0., 0.],
    ])
    expected = math.expm1(1.5) / math.expm1(3.0)
    assert penalty(env).tolist() == pytest.approx([expected] * 3)
    assert penalty(env).tolist() == pytest.approx([expected] * 3)
    advance(env)
    env.eef_wrench.zero_()  # A falling magnitude has the same cost as a rise.
    assert penalty(env).tolist() == pytest.approx([expected] * 3)


def test_wrench_curriculum_thresholds_keep_proportional_saturation(wrench_env):
    env, penalty, _ = wrench_env
    penalty(env, free_force_change=10.0, free_moment_change=1.0)
    advance(env)
    env.eef_wrench[:] = torch.tensor([
        [10., 0., 0., 1., 0., 0.],
        [17.5, 0., 0., 0., 0., 0.],
        [25., 0., 0., 2.5, 0., 0.],
    ])
    expected = [0.0, math.expm1(1.5) / math.expm1(3.0), 1.0]
    assert penalty(env, free_force_change=10.0, free_moment_change=1.0).tolist() == pytest.approx(expected)


def test_wrench_partial_reset_isolates_same_step_cache_and_stale_history(wrench_env):
    env, penalty, _ = wrench_env
    penalty(env)
    advance(env)
    env.eef_wrench[:, 0] = 200.0
    assert penalty(env).tolist() == [1.0, 1.0, 1.0]
    penalty.reset(torch.tensor([1]))
    assert penalty(env).tolist() == [1.0, 0.0, 1.0]
    advance(env)
    env.eef_wrench.zero_()
    env.sensor_data_fresh[2] = False
    assert penalty(env).tolist() == [1.0, 0.0, 0.0]
    advance(env)
    env.sensor_data_fresh[:] = True
    env.eef_wrench[:, 0] = 200.0
    assert penalty(env).tolist() == [1.0, 1.0, 0.0]


def test_wrench_invalid_sample_is_bounded_then_breaks_history(wrench_env):
    env, penalty, _ = wrench_env
    penalty(env)
    advance(env)
    env.eef_wrench[:, 0] = torch.tensor([1e30, math.inf, math.nan])
    assert penalty(env).tolist() == [1.0, 1.0, 1.0]
    advance(env)
    env.eef_wrench.zero_()
    assert penalty(env).tolist() == [1.0, 0.0, 0.0]


def test_real_manager_preserves_earned_positive_budget_and_absolute_rate_cap(wrench_env):
    env, _, manager = wrench_env
    manager._episode_sums["positive"].zero_()
    env.current_positive_rate[:] = torch.tensor([0.0, 0.01, 20.0])
    manager.compute(env.step_dt)
    for step in range(1, 21):
        advance(env)
        env.eef_wrench[:, 0] = 200.0 * (step % 2)
        manager.compute(env.step_dt)
        assert torch.all(-manager._episode_sums["wrench_change"] <=
                         0.05 * manager._episode_sums["positive"] + 1e-7)
        assert torch.all(manager._step_reward[:, -1].abs() <= 0.1 + 1e-7)
    assert manager._episode_sums["wrench_change"][0] == 0.0
    assert manager._episode_sums["wrench_change"][2] == pytest.approx(-0.04)


@pytest.fixture
def acceleration_env(rewards, make_env):
    env = make_env(3)
    env.step_dt = 0.02
    env.common_step_counter = env._sim_step_counter = 0
    env.scene["object_collection"].data.object_com_lin_acc_w = torch.zeros(3, 4, 3)
    penalty = rewards.ObjectAccelerationPenalty(SimpleNamespace(params={}), env)
    return env, penalty


def test_acceleration_threshold_independent_per_object_costs_and_all_object_sum(acceleration_env):
    env, penalty = acceleration_env
    penalty(env)
    advance(env)
    acceleration = env.scene["object_collection"].data.object_com_lin_acc_w
    acceleration[:] = torch.tensor([
        [[.6, .8, 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[1.25, 0., 0.], [0., -1.5, 0.], [0., 0., 0.], [0., 0., 0.]],
        [[1e30, 0., 0.], [1e30, 0., 0.], [1e30, 0., 0.], [1e30, 0., 0.]],
    ])
    expected = [0.0, 0.5 * (math.expm1(.5) + math.expm1(1.0)), 400.0]
    assert penalty(env).tolist() == pytest.approx(expected, rel=1e-6, abs=1e-6)
    env.target_id[:] = 3
    assert penalty(env).tolist() == pytest.approx(expected, rel=1e-6, abs=1e-6)
    advance(env)
    acceleration.zero_()
    assert penalty(env).tolist() == [0., 0., 0.]


@pytest.mark.parametrize("step_dt", [.01, .02, .1])
def test_acceleration_uses_physx_vector_without_control_dt_or_velocity_difference(acceleration_env, step_dt):
    env, penalty = acceleration_env
    env.step_dt = step_dt
    # Deliberately unusable velocity: only the framework acceleration is read.
    env.scene["object_collection"].data.object_lin_vel_w = torch.full((3, 4, 3), math.nan)
    penalty(env)
    advance(env)
    env.scene["object_collection"].data.object_com_lin_acc_w[:, 0] = torch.tensor([1.02, -1.36, 0.])
    expected = 0.5 * math.expm1(2.0 * .7)
    assert penalty(env).tolist() == pytest.approx([expected] * 3)
    assert not hasattr(penalty, "_previous_velocity")


def test_acceleration_partial_reset_same_step_cache_and_nonfinite_cap(acceleration_env):
    env, penalty = acceleration_env
    acceleration = env.scene["object_collection"].data.object_com_lin_acc_w
    penalty(env)
    advance(env)
    acceleration[:, :, 0] = 1e30
    assert penalty(env).tolist() == pytest.approx([400., 400., 400.])
    penalty.reset(torch.tensor([1]))
    assert penalty(env).tolist() == pytest.approx([400., 0., 400.])
    advance(env)
    acceleration[2, 0, 0] = math.nan
    acceleration[2, 1, 0] = math.inf
    acceleration[2, 2, 0] = -math.inf
    assert penalty(env).tolist() == pytest.approx([400., 0., 400.])
    advance(env)
    acceleration.zero_()
    assert penalty(env).tolist() == [0., 0., 0.]  # No stale velocity/history contaminates recovery.


def test_hand_contact_inclusive_net_norm_freshness_and_five_second_occupancy(rewards, make_env, monkeypatch):
    env = make_env(6)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(
        params={"position_tolerance": .01},
    ))
    env.force = torch.tensor([
        [.006, .008, 0.], [.009, 0., 0.], [1., 0., 0.],
        [math.nan, 0., 0.], [math.inf, 0., 0.], [1., 0., 0.],
    ])
    env.sensor_data_fresh[-1] = False
    monkeypatch.setattr(rewards, "whole_hand_contact_force_w", lambda env: env.force)
    assert rewards.hand_contact_any(env).tolist() == [1., 0., 1., 0., 0., 0.]
    assert sum(float(rewards.hand_contact_any(env)[2]) * .02 for _ in range(250)) == pytest.approx(5.0)


def test_hand_contact_fade_matches_live_goal_hold_tolerance_and_world_xy(rewards, make_env, monkeypatch):
    env = make_env(6)
    env.scene["object_collection"].data.object_pos_w.zero_()
    goal = torch.zeros(6, 3)
    goal[:, 0] = torch.tensor([.0, .03, .0375, .045, .3, .015])
    goal[:, 2] = 1000.0  # Height does not enter goal_hold or the fade factor.
    requested_commands = []

    def get_command(name):
        requested_commands.append(name)
        return goal

    env.command_manager.get_command = get_command
    hold_cfg = SimpleNamespace(params={"position_tolerance": .03, "command_name": "live_goal"})
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: hold_cfg)
    monkeypatch.setattr(rewards, "whole_hand_contact_force_w", lambda env: torch.ones(6, 3))
    assert rewards.hand_contact_any(env).tolist() == pytest.approx([0., 0., .5, 1., 1., 0.], abs=1e-6)
    hold_cfg.params["position_tolerance"] = .01  # A later curriculum value applies immediately.
    assert rewards.hand_contact_any(env).tolist() == pytest.approx([0., 1., 1., 1., 1., 1/3], abs=1e-6)
    assert requested_commands == ["live_goal", "live_goal"]


def test_hand_contact_zero_width_disables_fade_but_preserves_contact_gates(rewards, make_env, monkeypatch):
    env = make_env(6)
    # No goal manager access is needed, including when the object is at its goal.
    env.reward_manager = None
    env.command_manager = None
    force = torch.tensor([[.01, 0., 0.], [.009, 0., 0.], [1., 0., 0.],
                          [math.nan, 0., 0.], [math.inf, 0., 0.], [1., 0., 0.]])
    env.sensor_data_fresh[-1] = False
    monkeypatch.setattr(rewards, "whole_hand_contact_force_w", lambda env: force)
    assert rewards.hand_contact_any(env, goal_contact_fade_width=0.0).tolist() == [1., 0., 1., 0., 0., 0.]


@pytest.mark.parametrize("width", [-.01, math.nan, math.inf, -math.inf])
def test_hand_contact_invalid_fade_width_rejected(rewards, make_env, width):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rewards.hand_contact_any(make_env(1), goal_contact_fade_width=width)


def test_acceleration_cost_is_symmetric_for_speeding_up_and_slowing_down(acceleration_env):
    env, penalty = acceleration_env
    acceleration = env.scene["object_collection"].data.object_com_lin_acc_w
    penalty(env)
    advance(env)
    acceleration[0, 0, 0] = 1.7
    acceleration[1, 0, 0] = -1.7
    expected = .5 * math.expm1(1.4)
    assert penalty(env).tolist() == pytest.approx([expected, expected, 0.])


def test_acceleration_ignores_inactive_slots_and_does_not_keep_their_history(rewards, make_env):
    env = make_env(3)
    env.step_dt = .02
    env.common_step_counter = env._sim_step_counter = 0
    acceleration = torch.zeros(3, 7, 3)
    env.scene["object_collection"].data.object_com_lin_acc_w = acceleration
    env._active_object_mask = torch.zeros(3, 7, dtype=torch.bool)
    env._active_object_mask[:, 0] = True
    env._active_object_mask[2, 6] = True
    penalty = rewards.ObjectAccelerationPenalty(SimpleNamespace(params={}), env)
    penalty(env)
    advance(env)
    acceleration[~env._active_object_mask] = math.nan
    acceleration[1, 0, 0] = 1.7
    acceleration[2, 6, 0] = 1.7
    expected = .5 * math.expm1(1.4)
    assert penalty(env).tolist() == pytest.approx([0., expected, expected])
    assert torch.equal(penalty._previous_valid, env._active_object_mask)
    advance(env)
    acceleration.zero_()
    env._active_object_mask[0, 1] = True
    acceleration[0, 1] = torch.tensor([100., 0., 0.])
    assert penalty(env).tolist() == [0., 0., 0.]  # Activation gets a fresh first sample.
    advance(env)
    assert penalty(env).tolist() == pytest.approx([100., 0., 0.])


def test_obstacle_force_exponential_boundary_cap_and_stale_suppression(rewards, make_env, monkeypatch):
    env = make_env(5)
    env.force = torch.tensor([
        [12., 16., 0.], [20.5, 0., 0.], [1e30, 0., 0.],
        [math.nan, 0., 0.], [100., 0., 0.],
    ])
    env.sensor_data_fresh[-1] = False
    monkeypatch.setattr(rewards, "hand_obstacle_contact_force_w", lambda env: env.force)
    expected = [0., .5 * math.expm1(1.0), 100., 100., 0.]
    assert rewards.obstacle_contact_force_cost(env).tolist() == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    {"free_force_change": 0}, {"free_moment_change": math.nan},
    {"saturation_ratio": 1}, {"sharpness": 0}, {"max_positive_fraction": .051},
])
def test_wrench_invalid_parameters_rejected(wrench_env, kwargs):
    env, penalty, _ = wrench_env
    with pytest.raises(ValueError):
        penalty(env, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"free_acceleration": -.1}, {"free_acceleration": math.nan}, {"max_cost": 0},
])
def test_acceleration_invalid_parameters_rejected(acceleration_env, kwargs):
    env, penalty = acceleration_env
    with pytest.raises(ValueError):
        penalty(env, **kwargs)
