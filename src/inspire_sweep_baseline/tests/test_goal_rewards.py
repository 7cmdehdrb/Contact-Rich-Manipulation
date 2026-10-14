"""Behavioral checks for the new goal reward plan, without starting Isaac Sim."""

import math
from types import SimpleNamespace

import pytest
import torch


def _reach_term(rewards, env, near_radius=0.10):
    cfg = SimpleNamespace(params={"near_radius": near_radius, "frame_name": "ee_frame"})
    return rewards.ReachProgress(cfg, env)


def _set_shelf_quaternion(env, quaternion=None):
    env.scene["shelf"].data.root_quat_w = (
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(env.num_envs, -1).clone()
        if quaternion is None else quaternion
    )


def test_goal_cost_is_live_planar_isotropic_and_zero_only_at_goal(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(4)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    goal = position.clone()
    goal[:, :2] += torch.tensor([[0.1, 0.0], [0.0, -0.1], [-0.06, 0.08], [0.0, 0.0]])
    goal[:, 2] += 3.0  # Height is deliberately absent from goal shaping.
    env.command_manager.get_command = lambda name: goal
    expected = torch.tensor([math.expm1(-1.0)] * 3 + [0.0])
    assert torch.allclose(rewards.planar_goal_cost(env), expected)
    position[0, :2] = goal[0, :2]
    assert rewards.planar_goal_cost(env)[0] == 0.0
    position[1, 1] -= 0.05
    assert rewards.planar_goal_cost(env)[1] > expected[1]
    with pytest.raises(ValueError, match="sigma"):
        rewards.planar_goal_cost(env, sigma=0.0)


def test_action_groups_use_independent_normalized_means_and_applied_hand_clipping(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(3)
    current, previous = env.action_manager.action, env.action_manager.prev_action
    current[0, :6] = 2.0
    current[1, 6:] = torch.tensor([2.0, -1.0])
    previous[1, 6:] = torch.tensor([1.0, 0.0])
    current[2, 6:] = torch.tensor([1.0, 0.0])
    previous[2, 6:] = torch.tensor([0.0, 1.0])
    assert torch.equal(rewards.normalized_action_rate(env, "arm"), torch.tensor([4.0, 0.0, 0.0]))
    assert torch.equal(rewards.normalized_action_rate(env, "hand"), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(rewards.normalized_action_rate(env, "arm", 2.0), torch.tensor([1.0, 0.0, 0.0]))
    previous[0, :6] = current[0, :6]
    assert rewards.normalized_action_rate(env, "arm").count_nonzero() == 0
    # Holding a fully closed absolute Hand command has no posture penalty.
    current[:, 6:] = previous[:, 6:] = 0.0
    assert rewards.normalized_action_rate(env, "hand").count_nonzero() == 0
    with pytest.raises(ValueError, match="action_group"):
        rewards.normalized_action_rate(env, "fingers")


def test_joint_speed_cost_uses_resolved_group_only_and_per_joint_references(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(2)
    cfg = SimpleNamespace(name="robot", joint_ids=[2, 7])
    velocity = env.scene["robot"].data.joint_vel
    velocity[:, 0] = 100.0  # An excluded joint cannot affect the selected group.
    velocity[0, [2, 7]] = torch.tensor([1.0, -2.0])
    velocity[1, [2, 7]] = torch.tensor([0.0, 2.0])
    assert torch.equal(rewards.normalized_joint_velocity(env, cfg), torch.tensor([2.5, 2.0]))
    assert torch.equal(rewards.normalized_joint_velocity(env, cfg, [1.0, 2.0]), torch.tensor([1.0, 0.5]))
    with pytest.raises(ValueError, match="positive"):
        rewards.normalized_joint_velocity(env, cfg, 0.0)
    with pytest.raises(ValueError, match="group size"):
        rewards.normalized_joint_velocity(env, cfg, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.05])
def test_reaching_only_new_best_progress_is_bounded_and_dt_independent(load_mdp, make_env, step_dt):
    rewards = load_mdp("goal_rewards")
    env = make_env(1)
    env.step_dt = step_dt
    initial = env.scene["object_collection"].data.object_pos_w[0, 0].clone()
    reference = env.scene["ee_frame"].data.target_pos_w
    reference[0, 0] = initial + torch.tensor([1.1, 0.0, 0.0])
    term = _reach_term(rewards, env)
    assert term(env).item() == 0.0
    received = []
    for distance in (0.9, 1.0, 0.9, 0.5, 0.1, 0.0):
        reference[0, 0] = initial + torch.tensor([distance, 0.0, 0.0])
        received.append((term(env) * step_dt).item())
    assert received == pytest.approx([0.2, 0.0, 0.0, 0.4, 0.4, 0.0], abs=1.0e-6)
    assert sum(received) == pytest.approx(1.0)
    # Once the near region is reached, moving or waiting earns no more progress.
    reference[0, 0] = initial + torch.tensor([-0.05, 0.0, 0.0])
    assert term(env).item() == 0.0


def test_reaching_keeps_initial_object_position_when_live_object_moves(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(1)
    env.step_dt = 0.02
    initial = env.scene["object_collection"].data.object_pos_w[0, 0].clone()
    reference = env.scene["ee_frame"].data.target_pos_w
    reference[0, 0] = initial + torch.tensor([0.5, 0.0, 0.0])
    term = _reach_term(rewards, env)
    env.scene["object_collection"].data.object_pos_w[0, 0, 0] += 0.5
    assert term(env).item() == 0.0
    reference[0, 0, 0] -= 0.2
    assert (term(env) * env.step_dt).item() == pytest.approx(0.5)


def test_reach_partial_reset_reads_new_scene_before_command_reset_and_counts_first_step(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(3)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    reference = env.scene["ee_frame"].data.target_pos_w[:, 0]
    reference[:] = position + torch.tensor([0.6, 0.0, 0.0])
    term = _reach_term(rewards, env)
    reference[:] = position + torch.tensor([0.4, 0.0, 0.0])
    assert torch.allclose(term(env) * env.step_dt, torch.full((3,), 0.4))
    position[1] += torch.tensor([2.0, 3.0, 4.0])
    reference[1] = position[1] + torch.tensor([1.1, 0.0, 0.0])
    # Command still belongs to the previous episode when RewardManager resets.
    env.command_manager.get_command = lambda name: torch.full((3, 7), -123.0)
    term.reset(torch.tensor([1]))
    reference[1] = position[1] + torch.tensor([0.9, 0.0, 0.0])
    result = term(env) * env.step_dt
    assert result.tolist() == pytest.approx([0.0, 0.2, 0.0], abs=1.0e-6)


def test_reach_reset_refreshes_kinematics_before_reading_fixed_frame(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(1)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[0, 0]
    reference = env.scene["ee_frame"].data.target_pos_w
    reference[0, 0] = position + torch.tensor([99.0, 0.0, 0.0])

    class RobotData:
        @property
        def body_link_pos_w(self):
            reference[0, 0] = position + torch.tensor([0.6, 0.0, 0.0])
            return reference

    env.scene["robot"].data = RobotData()
    term = _reach_term(rewards, env)
    reference[0, 0] = position + torch.tensor([0.4, 0.0, 0.0])
    assert (term(env) * env.step_dt).item() == pytest.approx(0.4)


def test_reach_zero_initial_distance_is_finite_and_has_no_unearned_reward(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(2)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    env.scene["ee_frame"].data.target_pos_w[:, 0] = position
    term = _reach_term(rewards, env)
    assert torch.equal(term(env), torch.zeros(2))
    env.scene["ee_frame"].data.target_pos_w[:, 0, 0] += 0.2
    assert torch.equal(term(env), torch.zeros(2))
    env.scene["ee_frame"].data.target_pos_w[:, 0] = position
    assert torch.equal(term(env), torch.zeros(2))


def test_tilt_dead_zone_saturation_yaw_invariance_and_current_shelf_normal(load_mdp, make_env, tensor_math):
    rewards = load_mdp("goal_rewards")
    env = make_env(5)
    _set_shelf_quaternion(env)
    free, fail = math.radians(5.0), 0.9
    angles = torch.tensor([0.0, free, (free + fail) / 2.0, fail, math.pi])
    zeros = torch.zeros(5)
    env.scene["object_collection"].data.object_link_state_w[:, 0, 3:7] = tensor_math.quat_from_euler_xyz(
        angles, zeros, torch.linspace(0.0, math.pi, 5)
    )
    assert torch.allclose(rewards.object_tilt_angle(env), angles, atol=1.0e-6)
    expected = torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0])
    assert torch.allclose(rewards.object_tilt_cost(env, free, fail), expected, atol=1.0e-6)
    # Matching object and shelf rotations stay upright relative to the surface.
    shelf_quaternion = tensor_math.quat_from_euler_xyz(torch.full((5,), 0.3), zeros, zeros)
    _set_shelf_quaternion(env, shelf_quaternion)
    env.scene["object_collection"].data.object_link_state_w[:, 0, 3:7] = shelf_quaternion
    # float32 acos is sensitive near one; the residual stays far inside 5 degrees.
    assert torch.allclose(rewards.object_tilt_angle(env), zeros, atol=5.0e-4)
    assert rewards.object_tilt_cost(env, free, fail).count_nonzero() == 0
    with pytest.raises(ValueError, match="Tilt thresholds"):
        rewards.object_tilt_cost(env, fail, free)


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.05])
def test_terminal_events_pay_full_weight_and_union_failures_without_timeout_cost(load_mdp, make_env, step_dt):
    rewards = load_mdp("goal_rewards")
    env = make_env(4)
    env.step_dt = step_dt
    dones = {
        "success": torch.tensor([True, False, False, False]),
        "object_drop": torch.tensor([False, True, False, False]),
        "push_fast": torch.tensor([False, True, True, False]),
        "time_out": torch.tensor([False, False, False, True]),
    }
    env.termination_manager = SimpleNamespace(get_term=lambda name: dones[name])
    success = rewards.success_event(env) * 30.0 * step_dt
    failure = rewards.failure_event(env, ("object_drop", "push_fast")) * -80.0 * step_dt
    assert success.tolist() == pytest.approx([30.0, 0.0, 0.0, 0.0])
    assert failure.tolist() == pytest.approx([0.0, -80.0, -80.0, 0.0])
    # The terminal rows reset before the next transition's termination compute.
    for done in dones.values():
        done[:] = False
    assert rewards.success_event(env).count_nonzero() == 0
    assert rewards.failure_event(env, ("object_drop", "push_fast")).count_nonzero() == 0
    with pytest.raises(ValueError, match="exclude"):
        rewards.failure_event(env, ("time_out",))


def test_dense_costs_accumulate_per_second_without_extra_dt(load_mdp, make_env):
    rewards = load_mdp("goal_rewards")
    env = make_env(1)
    for step_dt in (0.01, 0.02, 0.05):
        assert (rewards.time_cost(env) * -0.1 * step_dt * round(1.0 / step_dt)).item() == pytest.approx(-0.1)
