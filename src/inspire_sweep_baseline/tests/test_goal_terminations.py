"""CPU checks for supported object-goal success and omni sweep failures."""

import math
from types import SimpleNamespace

import pytest
import torch


def _goal_env(make_env, count=3):
    """Add the live shelf fields needed by production termination functions."""
    env = make_env(count)
    env.step_dt = 0.02
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    position[:] = torch.tensor([0.0, 0.0, 1.05])
    shelf = env.scene["shelf"].data
    shelf.root_pos_w = torch.zeros(count, 3)
    shelf.root_quat_w = shelf.default_root_state[:, 3:7].clone()
    shelf.root_vel_w = torch.zeros(count, 6)
    goal = position.clone()
    env.command_manager.get_command = lambda name: goal
    return env, goal


def _success_term(terminations, env):
    return terminations.StableGoalSuccess(SimpleNamespace(params={}), env)


def _set_tilt(env, tensor_math, angles, yaw=None):
    count = env.num_envs
    angles = torch.as_tensor(angles).expand(count)
    zeros = torch.zeros(count)
    yaw = zeros if yaw is None else torch.as_tensor(yaw).expand(count)
    env.scene["object_collection"].data.object_link_state_w[:, 0, 3:7] = (
        tensor_math.quat_from_euler_xyz(angles, zeros, yaw)
    )


def test_success_requires_fifteen_consecutive_steps_and_resets_only_selected_rows(load_mdp, make_env):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env)
    success = _success_term(terminations, env)
    for _ in range(14):
        assert not success(env).any()
    assert success(env).tolist() == [True, True, True]
    success.reset(torch.tensor([1]))
    assert success(env).tolist() == [True, False, True]
    for _ in range(13):
        assert success(env).tolist() == [True, False, True]
    assert success(env).tolist() == [True, True, True]
    success.reset()
    assert not success(env).any()


@pytest.mark.parametrize("step_dt,required_steps", [(0.01, 30), (0.02, 15), (0.05, 6)])
def test_success_hold_duration_uses_control_time(load_mdp, make_env, step_dt, required_steps):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env, 1)
    env.step_dt = step_dt
    success = _success_term(terminations, env)
    for _ in range(required_steps - 1):
        assert not success(env).item()
    assert success(env).item()


@pytest.mark.parametrize("cause", ["position", "speed", "upright", "support", "footprint"])
def test_any_nonqualifying_step_clears_previous_success_progress(load_mdp, make_env, tensor_math, cause):
    terminations = load_mdp("goal_terminations")
    env, goal = _goal_env(make_env, 1)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    velocity = env.scene["object_collection"].data.object_lin_vel_w[:, 0]
    success = _success_term(terminations, env)
    for _ in range(14):
        assert not success(env).item()
    if cause == "position":
        position[0, 0] = 0.011
    elif cause == "speed":
        velocity[0, 1] = 0.021
    elif cause == "upright":
        _set_tilt(env, tensor_math, math.radians(11.0))
    elif cause == "support":
        position[0, 2] += 0.02
    elif cause == "footprint":
        position[0, 0] = goal[0, 0] = 0.161
    assert not success(env).item()
    position[:] = goal[:] = torch.tensor([0.0, 0.0, 1.05])
    velocity.zero_()
    _set_tilt(env, tensor_math, 0.0)
    for _ in range(14):
        assert not success(env).item()
    assert success(env).item()


def test_goal_transit_high_speed_and_strict_error_boundaries_do_not_succeed(load_mdp, make_env):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env, 5)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    velocity = env.scene["object_collection"].data.object_lin_vel_w[:, 0]
    position[0, 0] = 0.01  # Position tolerance is strict.
    position[1, 0] = 0.009
    velocity[1, 0] = 0.02  # Speed tolerance is also strict.
    velocity[2, :2] = torch.tensor([0.019, 0.019])  # Norm, not component cutoff.
    velocity[3, 0] = 0.30  # Crossing the exact goal does not confer success.
    position[4, :2] = torch.tensor([0.006, 0.006])
    success = _success_term(terminations, env)
    for _ in range(15):
        result = success(env)
    assert result.tolist() == [False, False, False, False, True]


def test_success_needs_support_and_full_footprint_but_allows_yaw(load_mdp, make_env, tensor_math):
    terminations = load_mdp("goal_terminations")
    env, goal = _goal_env(make_env, 7)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    position[0, 2] += 0.02  # Floating over the target.
    position[1, 2] -= 0.02  # Below the surface.
    position[2, 0] = 0.161  # Origin is on the board, but the cylinder overhangs.
    position[3, 1] = 0.461
    position[4, 0] = 0.159
    position[5, 1] = 0.459
    goal[:] = position
    _set_tilt(env, tensor_math, 0.0, torch.linspace(0.0, 2.0 * math.pi, 7))
    success = _success_term(terminations, env)
    for _ in range(15):
        result = success(env)
    assert result.tolist() == [False, False, False, False, True, True, True]


def test_supported_tilt_uses_lowest_bottom_rim_and_current_shelf_frame(load_mdp, make_env, tensor_math):
    terminations = load_mdp("goal_terminations")
    env, goal = _goal_env(make_env, 3)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    shelf = env.scene["shelf"].data
    tilt = math.radians(9.0)
    _set_tilt(env, tensor_math, tilt)
    position[0, 2] += 0.04 * math.sin(tilt)
    # The origin would pass a simple height check, while its lowest rim does not.
    position[1, 2] -= 0.006
    # A moved, yawed shelf must carry its valid support region with it.
    shelf.root_pos_w[2] = torch.tensor([2.0, -3.0, 0.2])
    shelf.root_quat_w[2] = tensor_math.quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2.0])
    )[0]
    position[2] = shelf.root_pos_w[2] + torch.tensor([-0.30, 0.10, 1.05])
    env.scene["object_collection"].data.object_link_state_w[2, 0, 3:7] = shelf.root_quat_w[2]
    goal[:] = position
    success = _success_term(terminations, env)
    for _ in range(15):
        result = success(env)
    assert result.tolist() == [True, False, True]


def test_each_failure_dominates_a_qualified_goal_and_timeout_is_not_a_failure(load_mdp, make_env):
    terminations = load_mdp("goal_terminations")
    failure_names = ("object_drop", "push_fast", "shelf_collision", "hand_velocity", "object_outside", "shelf_displacement")
    env, _ = _goal_env(make_env, len(failure_names) + 1)
    dones = {name: torch.zeros(env.num_envs, dtype=torch.bool) for name in failure_names + ("time_out",)}
    env.termination_manager = SimpleNamespace(get_term=lambda name: dones[name])
    success = _success_term(terminations, env)
    for _ in range(14):
        assert not success(env, failure_terms=failure_names).any()
    for row, name in enumerate(failure_names):
        dones[name][row] = True
    dones["time_out"][-1] = True
    assert success(env, failure_terms=failure_names).tolist() == [False] * len(failure_names) + [True]
    for name in failure_names:
        dones[name].zero_()
    for _ in range(14):
        assert not success(env, failure_terms=failure_names)[:-1].any()
    assert success(env, failure_terms=failure_names).all()
    # Only explicitly listed flags are consulted by the success term.
    success.reset()
    assert not success(env, failure_terms=failure_names + ("time_out",), hold_time=0.0)[-1]


def test_fall_and_tilt_preserve_height_cutoff_and_use_surface_normal(load_mdp, make_env, tensor_math):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env, 6)
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    env.scene.env_origins[:, 2] = torch.arange(6.0) * 2.0
    position[:, 2] = env.scene.env_origins[:, 2] + torch.tensor([1.039, 1.04, 1.05, 1.05, 1.05, 1.05])
    _set_tilt(env, tensor_math, torch.tensor([0.0, 0.0, 0.899, 0.901, 0.0, 0.3]), math.pi)
    env.scene["shelf"].data.root_quat_w[5] = env.scene["object_collection"].data.object_link_state_w[5, 0, 3:7]
    assert terminations.object_fall_or_tilt(env).tolist() == [True, False, False, True, False, False]
    # The computed angle itself must fail at equality with the configured cutoff.
    measured = terminations.object_tilt_angle(env)[2].item()
    assert terminations.object_fall_or_tilt(env, fail_angle=measured)[2]


def test_shelf_displacement_respects_origins_and_three_dimensional_five_cm_distance(load_mdp, make_env):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env, 5)
    shelf = env.scene["shelf"].data
    env.scene.env_origins[:] = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, -20.0, 3.0], [2.0, 4.0, 6.0], [9.0, 8.0, 7.0]])
    shelf.default_root_state[:, :3] = torch.tensor([-0.7, 0.0, 0.0])
    shelf.root_pos_w[:] = shelf.default_root_state[:, :3] + env.scene.env_origins
    shelf.root_pos_w[1, 1] += 0.05
    shelf.root_pos_w[2, 2] += 0.051
    shelf.root_pos_w[3, :2] += torch.tensor([0.04, 0.04])
    assert terminations.shelf_displacement(env).tolist() == [False, False, True, True, False]


def test_object_outside_shelf_uses_live_translation_rotation_and_board_bounds(load_mdp, make_env, tensor_math):
    terminations = load_mdp("goal_terminations")
    env, _ = _goal_env(make_env, 6)
    shelf = env.scene["shelf"].data
    shelf.root_pos_w[:] = torch.tensor([2.0, -3.0, 0.2])
    shelf.root_quat_w[:] = tensor_math.quat_from_euler_xyz(
        torch.zeros(6), torch.zeros(6), torch.full((6,), math.pi / 2.0)
    )
    local = torch.tensor([[0.199, 0.499, 1.05], [0.201, 0.0, 1.05], [-0.201, 0.0, 1.05], [0.0, 0.501, 1.05], [0.0, -0.501, 1.05], [0.0, 0.30, 1.05]])
    position = env.scene["object_collection"].data.object_pos_w[:, 0]
    position[:] = shelf.root_pos_w + tensor_math.quat_apply(shelf.root_quat_w, local)
    assert terminations.object_outside_shelf(env).tolist() == [False, True, True, True, True, False]
    # Moving just the shelf updates the containment result immediately.
    shelf.root_pos_w[5, 1] += 0.30
    assert terminations.object_outside_shelf(env)[5]


def test_shelf_collision_optional_fingers_keeps_wrist_and_motion(load_mdp, make_env):
    terminations = load_mdp("terminations")
    env, _ = _goal_env(make_env, 4)
    env.scene["wrist_frame"].data.target_pos_w[:, 0, 2] = 1.2
    env.scene["finger_frame"] = SimpleNamespace(data=SimpleNamespace(
        target_pos_w=torch.zeros(4, 2, 3)))
    env.scene["wrist_frame"].data.target_pos_w[1, 0, 2] = 1.10
    env.scene["shelf"].data.root_vel_w[2, 0] = 0.101
    env.scene["shelf"].data.root_vel_w[3, 5] = 0.101
    assert terminations.shelf_collision(env).tolist() == [True, True, True, True]
    assert terminations.shelf_collision(env, check_fingers=False).tolist() == [False, True, True, True]
    del env.scene["finger_frame"]
    assert terminations.shelf_collision(env, check_fingers=False).tolist() == [False, True, True, True]
