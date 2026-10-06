"""Push-v1 production command/reward regressions using real CPU tensors."""

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_contact_terms import _load, TEST_PACKAGE
from test_push_commands import _commands_fixture, _realize_reset


def _fixture(count=2, *, height=.01, preferred_roll=math.pi / 6, centered=False):
    env, old_command, _ = _commands_fixture(count)
    math_tools = sys.modules[f"{TEST_PACKAGE}.action_math"]
    env.cfg.task.contact_palm_height_m = height
    if centered:
        env.cfg.task.table_size = (.36, 1., .04)
        env.cfg.task.object_xy_range_low = (-.73, -.03)
        env.cfg.task.object_xy_range_high = (-.67, .03)
        env.cfg.task.command_centered_path = True
        env.cfg.task.command_midpoint_x_offset_m = .05
        env.cfg.task.command_initial_x_jitter_m = .002
    preferred = math_tools.quaternion_from_rotation_vector(torch.tensor((0., -math.pi / 2 + preferred_roll, 0.)))
    env.eef_quaternion[:] = preferred
    offset = math_tools.quaternion_from_rotation_vector(torch.tensor((.2, -.3, .15)))
    safe = SimpleNamespace(
        desired_c_quat_w=math_tools.quaternion_multiply(env.eef_quaternion, offset),
        c_quat_h=offset,
        palm_reference_pose_w=lambda: (env.eef_position, env.eef_quaternion),
    )
    env.event_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=safe))
    env.action_manager.action = torch.zeros(count, 8)
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    with patch.dict(sys.modules, {"isaaclab.utils": utils}):
        commands = _load(f"{TEST_PACKAGE}.mdp.push_v1_commands", "mdp/push_v1_commands.py")
    rewards = _load(f"{TEST_PACKAGE}.mdp.push_v1_rewards", "mdp/push_v1_rewards.py")
    command = commands.CubePushV1Command(old_command.cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    if not centered:
        command.set_reset_specs(torch.arange(count), [-.70, 0., 1.08], 0., .25)
    # Use the normal collision-safe reset height, not the new approach height.
    env.eef_position[:] = torch.tensor((-.70, -.14, 1.18))
    _realize_reset(env, command, torch.arange(count))
    env.scene["cube_table_contacts"].data.force_matrix_w_history[..., 2] = 9.81
    return env, command, rewards, math_tools, safe


def _step(env, command, counter):
    env.episode_length_buf[:] = 1
    env._sim_step_counter = counter
    return command.state()


def test_contact_approach_height_is_independent_of_safe_reset_and_rewards_descent():
    env, command, _, _, _ = _fixture()
    assert command.tracker.cfg.palm_height_offset_m == .01
    assert env.cfg.task.reset_position_offset_task[2] == .10
    # A palm on the old upper hover plane is still 9cm from the contact target.
    env.eef_position[:, 1] = -.03
    upper = _step(env, command, 2)
    torch.testing.assert_close(command.metrics["palm_height_error_m"], torch.full((2,), .09), atol=2.e-7, rtol=0)
    torch.testing.assert_close(command.metrics["approach_gap_m"], torch.full((2,), .09), atol=2.e-7, rtol=0)
    assert upper.approach_delta.min() > 0
    env.cfg.task.reset_position_offset_task = (.14, 0., .20)
    env.eef_position[:, 2] = 1.09
    lower = _step(env, command, 4)
    assert lower.approach_delta.min() > .8
    assert command.metrics["approach_gap_m"].max() < 2.e-7
    assert command.tracker.cfg.palm_height_offset_m == .01
    # Repeating a previous approach cannot farm the running-record reward.
    env.eef_position[:, 2] = 1.18
    assert not _step(env, command, 6).approach_delta.any()
    env.eef_position[:, 2] = 1.09
    assert not _step(env, command, 8).approach_delta.any()


def test_preferred_roll_recovers_selected_tilted_hand_pose_from_control_offset():
    env, command, rewards, math_tools, _ = _fixture(preferred_roll=math.pi / 6)
    preferred = math_tools.quaternion_rotate(env.eef_quaternion, torch.tensor((0., 0., 1.)).expand(2, -1))
    torch.testing.assert_close(command.preferred_finger_w, preferred, atol=2.e-7, rtol=0)
    assert preferred[0, 2] == pytest.approx(.5, abs=2.e-7)
    _step(env, command, 2)
    assert rewards.push_v1_roll_reward(env).abs().max() < 1.e-7
    roll_90 = math_tools.quaternion_from_rotation_vector(torch.tensor((0., math.pi / 2, 0.)))
    roll_180 = math_tools.quaternion_from_rotation_vector(torch.tensor((0., math.pi, 0.)))
    env.eef_quaternion[0] = math_tools.quaternion_multiply(env.eef_quaternion[0], roll_90)
    env.eef_quaternion[1] = math_tools.quaternion_multiply(env.eef_quaternion[1], roll_180)
    _step(env, command, 4)
    torch.testing.assert_close(rewards.push_v1_roll_reward(env), torch.tensor((-.5, -1.)), atol=3.e-7, rtol=0)
    # Local H+Y-axis roll preserves the normal: roll is a separate cost.
    assert rewards.push_v1_alignment_reward(env).abs().max() < 1.e-7


def test_full_normal_alignment_penalizes_downward_palm_before_contact():
    env, command, rewards, math_tools, _ = _fixture(preferred_roll=0)
    tilt = math_tools.quaternion_from_rotation_vector(torch.tensor((math.pi / 2, 0., 0.)))
    env.eef_quaternion[0] = tilt
    _step(env, command, 2)
    torch.testing.assert_close(rewards.push_v1_alignment_reward(env), torch.tensor((-.5, 0.)), atol=2.e-7, rtol=0)
    assert not command.metrics["raw_palm_contact"].any()
    assert (rewards.push_v1_roll_reward(env) <= 0).all()


def test_bilateral_roll_follows_each_sides_actual_selected_reset_pose():
    env, command, rewards, math_tools, safe = _fixture()
    left_roll = math_tools.quaternion_from_rotation_vector(torch.tensor((0., -math.pi / 2 - math.pi / 6, 0.)))
    reverse_normal = math_tools.quaternion_from_rotation_vector(torch.tensor((0., 0., math.pi)))
    env.eef_quaternion[1] = math_tools.quaternion_multiply(left_roll, reverse_normal)
    safe.desired_c_quat_w[1] = math_tools.quaternion_multiply(env.eef_quaternion[1], safe.c_quat_h)
    command.set_reset_specs([0], [-.70, 0., 1.08], 0., .25)
    command.set_reset_specs([1], [-.70, 0., 1.08], math.pi, .25)
    _realize_reset(env, command, torch.arange(2))
    forces = env.scene["cube_palm_contacts"].data.force_matrix_w_history
    forces[0, :, 0, 0, 1] = .02
    forces[1, :, 0, 0, 1] = -.02
    state = _step(env, command, 2)
    assert state.first_contact.tolist() == [True, True]
    torch.testing.assert_close(command.preferred_finger_w[:, 2], torch.tensor((.5, -.5)), atol=3.e-7, rtol=0)
    assert rewards.push_v1_alignment_reward(env).abs().max() < 1.e-7
    assert rewards.push_v1_roll_reward(env).abs().max() < 1.e-7


def test_diagnostics_distinguish_raw_contact_force_direction_and_support_and_keep_latches():
    env, command, _, _, _ = _fixture(4)
    forces = env.scene["cube_palm_contacts"].data.force_matrix_w_history
    forces[1, :, 0, 0, 1] = .02
    forces[2, :, 0, 0, 2] = .02
    forces[3, :, 0, 0, 1] = .02
    env.scene["cube_table_contacts"].data.force_matrix_w_history[3] = 0
    state = _step(env, command, 2)
    assert command.metrics["raw_palm_contact"].tolist() == [0., 1., 1., 1.]
    assert command.metrics["side_alignment_gate"].tolist() == [0., 1., 0., 1.]
    assert command.metrics["grounded"].tolist() == [1., 1., 1., 0.]
    assert state.first_contact.tolist() == [False, True, False, False]
    assert state.valid_push_contact.tolist() == [False, True, False, False]
    forces.zero_()
    _step(env, command, 4)
    assert not command.metrics["raw_palm_contact"].any()
    assert command.metrics["raw_contact_seen"].tolist() == [0., 1., 1., 1.]
    assert command.metrics["side_gate_seen"].tolist() == [0., 1., 0., 1.]
    torch.testing.assert_close(command.metrics["raw_contact_time_s"], torch.tensor((0., .02, .02, .02)))
    torch.testing.assert_close(command.metrics["valid_push_contact_time_s"], torch.tensor((0., .02, 0., 0.)))


def test_input_clipping_uses_manager_caller_values_not_processed_action_or_torque_flags():
    env, command, _, _, _ = _fixture(4)
    env.action_manager.action[1] = torch.tensor((2., -2., 1., 0., 0., 0., 3., 0.))
    env.action_manager.action[2, 0] = float("nan")
    env.action_manager.action[3] = -2
    _step(env, command, 2)
    torch.testing.assert_close(command.metrics["input_action_clip_fraction"], torch.tensor((0., 3/8, 0., 1.)))
    assert command.metrics["input_action_clip_any"].tolist() == [0., 1., 0., 1.]
    assert command.metrics["invalid_input_action"].tolist() == [0., 0., 1., 0.]


def test_new_contact_height_retains_grounded_side_contact_progress_gate_and_budget():
    env, command, _, _, _ = _fixture()
    cube = env.scene["target_object"].data.root_pos_w
    cube[:, 1] += .001
    assert not _step(env, command, 2).progress_delta.any()
    forces = env.scene["cube_palm_contacts"].data.force_matrix_w_history
    forces[:, :, 0, 0, 1] = .02
    assert not _step(env, command, 4).progress_delta.any()  # no retroactive drift credit
    cube[:, 1] += .001
    state = _step(env, command, 6)
    assert (state.progress_delta > .009).all()
    assert (state.progress_delta < .01).all()
    # Removing support still disables credit even with an aligned pad force.
    env.scene["cube_table_contacts"].data.force_matrix_w_history.zero_()
    cube[:, 1] += .001
    assert not _step(env, command, 8).progress_delta.any()


def test_shared_cache_and_partial_reset_do_not_double_count_contact_or_clear_other_rows():
    env, command, rewards, math_tools, safe = _fixture()
    env.scene["cube_palm_contacts"].data.force_matrix_w_history[:, :, 0, 0, 1] = .02
    flip = math_tools.quaternion_from_rotation_vector(torch.tensor((0., math.pi, 0.)))
    env.eef_quaternion[1] = math_tools.quaternion_multiply(env.eef_quaternion[1], flip)
    before = _step(env, command, 2)
    before_penalty = rewards.push_v1_roll_reward(env)
    with patch.object(command, "_snapshot", wraps=command._snapshot) as snapshot:
        assert command.state() is before
        command.compute(.02)
        assert command.state() is before
        rewards.push_v1_roll_reward(env)
        assert snapshot.call_count == 0
    torch.testing.assert_close(command.metrics["raw_contact_time_s"], torch.full((2,), .02))
    assert before.first_contact.tolist() == [True, True]
    torch.testing.assert_close(before_penalty, torch.tensor((0., -1.)), atol=2.e-7, rtol=0)

    # Geometry can choose a different safe roll for just the resetting row.
    selected = math_tools.quaternion_from_rotation_vector(torch.tensor((0., -math.pi / 4, 0.)))
    env.eef_quaternion[1] = selected
    safe.desired_c_quat_w[1] = math_tools.quaternion_multiply(selected, safe.c_quat_h)
    command.set_reset_specs([1], [-.70, 0., 1.08], 0., .25)
    _realize_reset(env, command, [1])
    env.episode_length_buf[1] = 0
    after = command.state()
    assert after.first_contact.tolist() == [True, False]
    assert command.state() is after
    assert command.metrics["raw_contact_seen"].tolist() == [1., 0.]
    torch.testing.assert_close(command.metrics["raw_contact_time_s"], torch.tensor((.02, 0.)))
    assert command.metrics["palm_height_error_m"][1] == 0
    assert rewards.push_v1_roll_reward(env)[1] == 0
    torch.testing.assert_close(before_penalty, torch.tensor((0., -1.)), atol=2.e-7, rtol=0)
    next_step = _step(env, command, 4)
    assert next_step.first_contact.tolist() == [False, True]
    torch.testing.assert_close(command.metrics["raw_contact_time_s"], torch.tensor((.04, .02)))
    assert rewards.push_v1_roll_reward(env).abs().max() < 1.e-7


def test_alignment_and_roll_costs_are_zero_on_failure_and_reset_rows():
    env, command, rewards, math_tools, _ = _fixture()
    env.eef_quaternion[:] = math_tools.quaternion_from_rotation_vector(torch.tensor((math.pi, 0., 0.)))
    env.scene["table_contacts"].data.force_matrix_w_history[0, 1, 0, 37, 0] = .01
    env.episode_length_buf[:] = torch.tensor((1, 0))
    env._sim_step_counter = 2
    state = command.state()
    assert state.failure.tolist() == [True, False]
    assert not rewards.push_v1_alignment_reward(env).any()
    assert not rewards.push_v1_roll_reward(env).any()
    assert not state.progress_delta.any()
    assert not state.approach_delta.any()


def test_action_excess_exact_formula_uses_raw_input_and_respects_selected_rows():
    env, _, rewards, _, _ = _fixture(4)
    env.action_manager.action[0] = torch.tensor((-1., 1., .5, -.5, 0., 0., 0., 0.))
    env.action_manager.action[1] = 2.
    env.action_manager.action[2] = torch.tensor((3., -2., 1., 0., 0., 0., 0., 0.))
    env.action_manager.action[3] = -3.
    torch.testing.assert_close(rewards.push_v1_action_excess_reward(env), torch.tensor((0., -1., -5/8, -4.)))
    env.action_manager.action[1] = 0
    torch.testing.assert_close(rewards.push_v1_action_excess_reward(env), torch.tensor((0., 0., -5/8, -4.)))


def test_action_excess_is_finite_for_invalid_and_extreme_float32_input():
    env, _, rewards, _, _ = _fixture(4)
    env.action_manager.action[0] = torch.tensor((float("nan"), float("inf"), -float("inf"), 2., -2., 0., 0., 0.))
    env.action_manager.action[1] = 10
    env.action_manager.action[2] = torch.finfo(torch.float32).max
    env.action_manager.action[3] = -torch.finfo(torch.float32).max
    value = rewards.push_v1_action_excess_reward(env)
    assert torch.isfinite(value).all()
    assert (value <= 0).all() and (value >= -1.e12).all()
    torch.testing.assert_close(value[:2], torch.tensor((-.25, -81.)))
    maximum_cost = -(torch.tensor(1.e6) - 1.).square()
    torch.testing.assert_close(value[2:], maximum_cost.expand(2))


def test_action_excess_is_a_rate_with_exactly_one_manager_dt_integration():
    env, _, rewards, _, _ = _fixture()
    env.action_manager.action[:] = 2.
    rate = rewards.push_v1_action_excess_reward(env)
    torch.testing.assert_close(rate, torch.full((2,), -1.))
    episode_cost = rate * .002 * .02 * 500
    env.step_dt = .01
    torch.testing.assert_close(rewards.push_v1_action_excess_reward(env), rate)
    torch.testing.assert_close(episode_cost, rate * .002 * .01 * 1000)
    torch.testing.assert_close(episode_cost, torch.full((2,), -.02))


def test_centered_sampler_keeps_once_uniform_angle_length_and_conditional_x_draws():
    count = 2048
    env, command, _, _, _ = _fixture(count, centered=True, height=.025)
    env.cfg.task.command_sample_attempts = 0  # the centered sampler never uses rejection retries
    env.scene.env_origins[:, 0] = (torch.arange(count) % 16) * 2.5
    env.scene.env_origins[:, 1] = (torch.arange(count) // 16) * 2.5
    env.base_position[:, :2] = env.scene.env_origins[:, :2]
    torch.manual_seed(814)
    expected_angle = (2. * torch.rand(count) - 1.) * (math.pi / 18)
    expected_angle += torch.randint(0, 2, (count,)) * math.pi
    expected_distance = torch.empty(count).uniform_(.2, .3)
    expected_direction = torch.stack((-expected_angle.sin(), expected_angle.cos(), torch.zeros(count)), -1)
    expected_jitter = (2. * torch.rand(count) - 1.) * .002
    expected_x = -.70 - .5 * expected_distance * expected_direction[:, 0] + expected_jitter
    expected_y = -.03 + torch.rand(count) * .06
    torch.manual_seed(814)
    position = command.sample_reset(torch.arange(count))
    torch.testing.assert_close(command.angle_rad, expected_angle, atol=0, rtol=0)
    torch.testing.assert_close(command.distance_m, expected_distance, atol=0, rtol=0)
    torch.testing.assert_close(position[:, 0], expected_x + env.scene.env_origins[:, 0], atol=4.e-6, rtol=0)
    torch.testing.assert_close(position[:, 1], expected_y + env.scene.env_origins[:, 1], atol=4.e-5, rtol=0)
    local = position - env.scene.env_origins
    assert (local[:, 0] >= -.728048 - 4.e-6).all() and (local[:, 0] <= -.671952 + 4.e-6).all()
    assert (local[:, 1].abs() <= .03 + 4.e-5).all()
    assert .45 < float((command.angle_rad.cos() > 0).float().mean()) < .55
    padding = math.sqrt(3.) * .03 + .01
    for fraction in torch.linspace(0, 1, 21):
        point = torch.lerp(command.target_pos_w, command.goal_pos_w, fraction) - env.scene.env_origins
        assert ((point[:, :2] - torch.tensor((-.75, 0.))).abs() <= torch.tensor((.18, .5)) - padding + 4.e-5).all()


def test_centered_sampler_all_boundary_fixtures_preserve_parent_path_validation_without_random_draws():
    import itertools
    corners = list(itertools.product((0., math.pi), (-math.pi / 18, math.pi / 18), (.2, .3), (-.002, .002), (-.03, .03)))
    env, command, _, _, _ = _fixture(len(corners), centered=True)
    angle = torch.tensor([side + jitter for side, jitter, _, _, _ in corners])
    distance = torch.tensor([length for _, _, length, _, _ in corners])
    direction = torch.stack((-angle.sin(), angle.cos(), torch.zeros(len(corners))), -1)
    position = torch.empty(len(corners), 3)
    position[:, 0] = -.70 - .5 * distance * direction[:, 0] + torch.tensor([jitter for _, _, _, jitter, _ in corners])
    position[:, 1] = torch.tensor([y for _, _, _, _, y in corners])
    position[:, 2] = 1.08
    command.set_reset_specs(torch.arange(len(corners)), position, angle, distance)
    with patch.object(torch, "rand", side_effect=AssertionError("fixed reset must not draw")):
        with patch.object(torch, "randint", side_effect=AssertionError("fixed reset must not draw")):
            result = command.sample_reset(torch.arange(len(corners)))
    torch.testing.assert_close(result, position)
    torch.testing.assert_close(command.angle_rad, angle)
    torch.testing.assert_close(command.distance_m, distance)
    padding = math.sqrt(3.) * .03 + .01
    for fraction in torch.linspace(0, 1, 21):
        point = torch.lerp(result, command.goal_pos_w, fraction)
        assert ((point[:, :2] - torch.tensor((-.75, 0.))).abs() <= torch.tensor((.18, .5)) - padding).all()


def test_centered_subset_sampling_keeps_unselected_commands_queued_specs_and_fixed_episode_goal():
    env, command, _, _, _ = _fixture(6, centered=True)
    buffers = (command.target_pos_w, command.goal_pos_w, command.angle_rad, command.distance_m, command.direction_w)
    before = [value.clone() for value in buffers]
    command.set_reset_specs([2], [-.70, .02, 1.08], math.pi / 18, .3)
    command.set_reset_specs([3], [-.70, -.02, 1.08], math.pi, .2)
    result = command.sample_reset([2, 5])
    torch.testing.assert_close(result[0], torch.tensor((-.70, .02, 1.08)))
    assert command.angle_rad[2] == pytest.approx(math.pi / 18)
    assert command.distance_m[2] == pytest.approx(.3)
    untouched = [0, 1, 3, 4]
    for value, initial in zip(buffers, before):
        torch.testing.assert_close(value[untouched], initial[untouched])
    assert command._specified[3] and not command._specified[2]
    env.scene["target_object"].data.root_pos_w[[2, 5]] = result
    command.reset([2, 5])
    fixed_target, fixed_goal = command.target_pos_w.clone(), command.goal_pos_w.clone()
    env.scene["target_object"].data.root_pos_w[:, 1] += .015
    command.compute(.02)
    torch.testing.assert_close(command.target_pos_w, fixed_target)
    torch.testing.assert_close(command.goal_pos_w, fixed_goal)
    assert command._specified[3]


@pytest.mark.parametrize("attribute,value", [
    ("object_xy_range_high", (-.71, .03)),
    ("command_midpoint_x_offset_m", .11),
    ("command_initial_x_jitter_m", .04),
    ("table_size", (.20, 1., .04)),
    ("object_xy_range_high", (-.67, .20)),
    ("command_initial_x_jitter_m", -.002),
])
def test_centered_misconfiguration_raises_before_draw_or_buffer_commit(attribute, value):
    env, command, _, _, _ = _fixture(3, centered=True)
    setattr(env.cfg.task, attribute, value)
    before = [tensor.clone() for tensor in (command.target_pos_w, command.goal_pos_w, command.angle_rad,
                                           command.distance_m, command._pending, command._specified)]
    with patch.object(torch, "rand", side_effect=AssertionError("invalid config must not bias random commands")):
        with pytest.raises(ValueError, match="Centered Push sampling"):
            command.sample_reset([0, 2])
    for tensor, initial in zip((command.target_pos_w, command.goal_pos_w, command.angle_rad,
                                command.distance_m, command._pending, command._specified), before):
        torch.testing.assert_close(tensor, initial)


def test_centered_draw_clearance_failure_is_explicit_and_atomic_without_retry():
    env, command, _, _, _ = _fixture(3, centered=True)
    before = [tensor.clone() for tensor in (command.target_pos_w, command.goal_pos_w, command.angle_rad,
                                           command.distance_m, command._pending, command._specified)]
    with patch.object(command, "_valid_path", return_value=torch.zeros(2, dtype=torch.bool)) as check:
        with pytest.raises(RuntimeError, match="no commands were committed"):
            command.sample_reset([0, 2])
        assert check.call_count == 1
    for tensor, initial in zip((command.target_pos_w, command.goal_pos_w, command.angle_rad,
                                command.distance_m, command._pending, command._specified), before):
        torch.testing.assert_close(tensor, initial)


@pytest.mark.parametrize("all_corners_at_env1927", (False, True), ids=("2048_training_grid", "32_corners_env1927"))
def test_centered_boundary_fixtures_accept_world_roundoff_on_large_training_origins(all_corners_at_env1927):
    import itertools
    count = 2048
    env, command, _, _, _ = _fixture(count, centered=True, height=.025)
    index = torch.arange(count)
    rows, columns = int(math.sqrt(count)), math.ceil(count / int(math.sqrt(count)))
    env.scene.env_origins[:, 0] = (.5 * (columns - 1) - torch.div(index, rows, rounding_mode="floor")) * 2.5
    env.scene.env_origins[:, 1] = (index % rows - .5 * (rows - 1)) * 2.5
    torch.testing.assert_close(env.scene.env_origins[1927], torch.tensor((-48.75, 37.5, 0.)))
    if all_corners_at_env1927:
        env.scene.env_origins[:32] = env.scene.env_origins[1927].clone()
    env.base_position[:, :2] = env.scene.env_origins[:, :2]
    corners = list(itertools.product((0., math.pi), (-math.pi / 18, math.pi / 18), (.2, .3), (-.002, .002), (-.03, .03)))
    cases = [corners[i % len(corners)] for i in range(count)]
    angle = torch.tensor([side + jitter for side, jitter, _, _, _ in cases])
    distance = torch.tensor([length for _, _, length, _, _ in cases])
    direction = torch.stack((-angle.sin(), angle.cos(), torch.zeros(count)), -1)
    position = torch.empty(count, 3)
    position[:, 0] = -.70 - .5 * distance * direction[:, 0] + torch.tensor([jitter for _, _, _, jitter, _ in cases])
    position[:, 1] = torch.tensor([y for _, _, _, _, y in cases])
    position[:, 2] = 1.08
    command.set_reset_specs(index, position, angle, distance)
    stored_world = command._specified_position.clone()
    reconstructed = stored_world - env.scene.env_origins
    # These valid local +/-3cm boundaries can differ by several micrometres
    # after world float32 storage, reproducing the old 1e-7 comparison failure.
    assert ((reconstructed[:, 1] > .03 + 1.e-7) | (reconstructed[:, 1] < -.03 - 1.e-7)).any()
    with patch.object(torch, "rand", side_effect=AssertionError("valid fixed specs must not be resampled")):
        result = command.sample_reset(index)
    torch.testing.assert_close(result, stored_world, atol=0, rtol=0)
    torch.testing.assert_close(command.angle_rad, angle, atol=0, rtol=0)
    torch.testing.assert_close(command.distance_m, distance, atol=0, rtol=0)
    padding = math.sqrt(3.) * .03 + .01
    for fraction in torch.linspace(0, 1, 11):
        point = torch.lerp(result, command.goal_pos_w, fraction) - env.scene.env_origins
        assert ((point[:, :2] - torch.tensor((-.75, 0.))).abs() <= torch.tensor((.18, .5)) - padding).all()


def test_centered_roundoff_tolerance_rejects_a_position_more_than_one_world_ulp_outside_bounds():
    env, command, _, _, _ = _fixture(2, centered=True)
    env.scene.env_origins[:] = torch.tensor((-48.75, 37.5, 0.))
    env.base_position[:, :2] = env.scene.env_origins[:, :2]
    command.set_reset_specs([0], [-.70, .03, 1.08], 0., .2)
    command.set_reset_specs([1], [-.70, -.03, 1.08], math.pi, .2)
    before = command.target_pos_w.clone()
    target = torch.tensor((float("inf"), -float("inf")))
    for _ in range(4):
        command._specified_position[:, 1] = torch.nextafter(command._specified_position[:, 1], target)
    with pytest.raises(RuntimeError, match="no commands were committed"):
        command.sample_reset([0, 1])
    torch.testing.assert_close(command.target_pos_w, before)
    assert command._specified.all()


@pytest.mark.parametrize("height", [-.001, float("nan"), float("inf")])
def test_invalid_contact_height_is_rejected_by_runtime_command_constructor(height):
    with pytest.raises(ValueError, match="contact_palm_height_m"):
        _fixture(height=height)
