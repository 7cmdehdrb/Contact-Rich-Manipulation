"""Root-frame polar goal semantics and separation from the moving target."""

import math
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, _module_from_file


@pytest.fixture
def observations(load_mdp):
    load_mdp("goal_events")  # Install the Isaac Lab CPU tensor-math harness.
    return _module_from_file(
        "_baseline_cpu_fusion_observations", MODULE_ROOT / "contact_sweep_clutter_fusion/observations.py",
    )


def test_action_feedback_is_bounded_without_mutating_raw_reward_inputs(observations):
    raw = torch.tensor([[.2, -.4, 2., -3., 1e6, -1e6, 9e5, -5e4]])
    previous = raw.clone() - .1
    env = SimpleNamespace(action_manager=SimpleNamespace(action=raw, prev_action=previous))
    raw_before, previous_before = raw.clone(), previous.clone()
    feedback = observations.bounded_last_action(env)
    torch.testing.assert_close(feedback, raw_before.clamp(-1, 1))
    feedback.zero_()
    torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
    torch.testing.assert_close(previous, previous_before, rtol=0, atol=0)


def test_bounded_action_feedback_breaks_amplifying_loop(observations):
    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.full((1, 8), .5)))
    for _ in range(100):
        feedback = observations.bounded_last_action(env)
        env.action_manager.action = 2 * feedback + .1
    assert torch.isfinite(env.action_manager.action).all()
    assert env.action_manager.action.abs().max() <= 2.1


def _set_command(env, displacements_w):
    initial = env.scene["object_collection"].data.object_link_state_w[:, 0].clone()
    command = SimpleNamespace(target_init_state_w=initial, command=initial[:, :7].clone())
    command.command[:, :3] += displacements_w
    env.command_manager.get_term = lambda name: command
    return command


def test_clockwise_cardinal_directions_use_initial_target_origin(observations, make_env):
    env = make_env(4)
    command = _set_command(env, torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0],
                                            [-0.3, 0.0, 0.0], [0.0, 0.4, 0.0]]))
    original_goal = command.command.clone()
    actual = observations.initial_goal_polar_in_root(env)
    assert actual.shape == (4, 2)
    assert torch.allclose(actual[:, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]), atol=1e-7)
    assert torch.allclose(actual[:, 1], torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2]))
    assert torch.equal(command.command, original_goal)
    # This checks the push distance, not the robot-to-goal distance (~0.75 m).
    assert not torch.allclose(actual[:, 0], original_goal[:, :2].norm(dim=-1))


@pytest.mark.parametrize("root_angle", [math.pi, -0.7, 0.2])
def test_root_rotation_and_translation_preserve_goal_axes(observations, make_env, tensor_math, root_angle):
    env = make_env(4)
    robot = env.scene["robot"].data
    robot.root_quat_w[:] = torch.tensor([math.cos(root_angle / 2), 0.0, 0.0, math.sin(root_angle / 2)])
    robot.root_pos_w[:] = torch.tensor([5.0, -10.0, 0.5])
    delta_root = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0],
                              [-0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    command = _set_command(env, tensor_math.quat_apply(robot.root_quat_w, delta_root))
    actual = observations.initial_goal_polar_in_root(env)
    assert torch.allclose(actual[:, 0], torch.full((4,), 0.1), atol=1e-7)
    expected_theta = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    # Compare periodic direction at the 0/2pi boundary under float32 rotation.
    assert torch.allclose(actual[:, 1].cos(), expected_theta.cos(), atol=1e-6)
    assert torch.allclose(actual[:, 1].sin(), expected_theta.sin(), atol=1e-6)
    command.command[:, :3] += torch.tensor([9.0, -1.0, 5.0])
    command.target_init_state_w[:, :3] += torch.tensor([9.0, -1.0, 5.0])
    translated = observations.initial_goal_polar_in_root(env)
    assert torch.allclose(translated[:, 0], actual[:, 0], atol=1e-6)
    assert torch.allclose(translated[:, 1].cos(), actual[:, 1].cos(), atol=1e-5)
    assert torch.allclose(translated[:, 1].sin(), actual[:, 1].sin(), atol=1e-5)


def test_live_target_cannot_leak_through_goal_observation(observations, make_env):
    env = make_env(3)
    command = _set_command(env, torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]))
    before = observations.initial_goal_polar_in_root(env)
    goal_before, initial_before = command.command.clone(), command.target_init_state_w.clone()
    env.scene["object_collection"].data.object_pos_w += torch.tensor([5.0, -3.0, 1.0])
    env.scene["ee_frame"].data.target_pos_w -= 1.0
    assert torch.equal(observations.initial_goal_polar_in_root(env), before)
    assert torch.equal(command.command, goal_before)
    assert torch.equal(command.target_init_state_w, initial_before)
    # Mutating an observation result also cannot mutate the command buffers.
    observations.initial_goal_polar_in_root(env).fill_(100.0)
    assert torch.equal(observations.initial_goal_polar_in_root(env), before)


def test_goal_angle_range_wraparound_zero_and_partial_command_reset(observations, make_env):
    env = make_env(4)
    command = _set_command(env, torch.tensor([[0.1, 1e-4, 0.0], [0.1, -1e-4, 0.0],
                                            [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]]))
    before = observations.initial_goal_polar_in_root(env)
    assert ((before[:, 1] >= 0.0) & (before[:, 1] < 2.0 * math.pi)).all()
    assert before[0, 1] > 2.0 * math.pi - 0.01
    assert before[1, 1] < 0.01
    assert torch.equal(before[2], torch.zeros(2))
    command.command[3, :3] = command.target_init_state_w[3, :3] + torch.tensor([-0.2, 0.0, 0.0])
    after = observations.initial_goal_polar_in_root(env)
    assert torch.equal(after[:3], before[:3])
    assert torch.allclose(after[3], torch.tensor([0.2, math.pi]))
