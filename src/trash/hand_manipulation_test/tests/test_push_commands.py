"""Real-tensor command sampling and base-frame observation contracts."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_contact_terms import _environment, _load, _load_terms, TEST_PACKAGE


def _commands_fixture(count=4):
    _, contact_commands, *_ = _load_terms()
    sys.modules[f"{TEST_PACKAGE}.mdp.contact_commands"] = contact_commands
    geometry = sys.modules[f"{TEST_PACKAGE}.geometry"]
    geometry.base_link_pose_w = lambda env: (env.base_position, env.base_quaternion)
    geometry.palm_tactile_bits = lambda env: torch.zeros(env.num_envs, 17)
    geometry.wrist_wrench_c = lambda env: None
    _load(f"{TEST_PACKAGE}.action_math", "action_math.py")
    _load(f"{TEST_PACKAGE}.push_state", "push_state.py")
    from types import ModuleType
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    with patch.dict(sys.modules, {"isaaclab.utils": utils}):
        commands = _load(f"{TEST_PACKAGE}.mdp.push_commands", "mdp/push_commands.py")
    observations = _load(f"{TEST_PACKAGE}.mdp.observations", "mdp/observations.py")
    env = _environment(count)
    env.step_dt = .02
    env.base_position = env.scene.env_origins.clone()
    env.base_position[:, 2] = .79505
    env.base_quaternion = torch.tensor((0., 0., 0., 1.)).expand(count, -1).clone()
    env.episode_length_buf.zero_()
    cube = env.scene["target_object"].data
    cube.root_pos_w[:] = torch.tensor((-.70, 0., 1.08))
    cube.root_quat_w = torch.tensor((1., 0., 0., 0.)).expand(count, -1).clone()
    cube.root_lin_vel_w = torch.zeros(count, 3)
    env.scene["table"] = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor((-.75, 0., 1.03)).expand(count, -1).clone(),
        root_quat_w=cube.root_quat_w.clone(),
    ))
    env.scene["cube_palm_contacts"].data.force_matrix_w_history = torch.zeros(count, 2, 1, 17, 3)
    env.scene["cube_table_contacts"] = SimpleNamespace(data=SimpleNamespace(
        force_matrix_w_history=torch.zeros(count, 2, 1, 1, 3)))
    env.cfg.task = SimpleNamespace(
        table_size=(.36, 1., .04), table_center=(-.75, 0., 1.03), cube_size=.06,
        command_distance_range=(.2, .3), command_angle_jitter_rad=math.pi/18,
        command_sample_attempts=128, object_xy_range_low=(-.71, -.06), object_xy_range_high=(-.69, .06),
        cube_center_height_m=1.08, table_path_margin_m=.01, table_failure_margin_m=.005,
        contact_threshold_n=.01, side_contact_angle_rad=math.pi/6, goal_reward_sigma_m=.05,
        progress_sigma_fraction=.5,
        success_distance_m=.01, success_speed_m_s=.02, success_hold_time_s=.2,
        reset_position_offset_task=(.14, 0., .10),
    )
    env.event_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=SimpleNamespace(
        palm_reference_pose_w=lambda: (env.eef_position, env.eef_quaternion))))
    cfg = SimpleNamespace(
        target_position_range_low=(-.71, -.06, 1.08), target_position_range_high=(-.69, .06, 1.08),
        resampling_time_range=(1.e6, 1.e6), debug_vis=False, asset_name="robot", object_name="target_object",
    )
    command = commands.CubePushCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    return env, command, observations


def _realize_reset(env, command, index):
    position = command.sample_reset(index)
    env.scene["target_object"].data.root_pos_w[index] = position
    command.reset(index)


def test_continuous_bilateral_sampling_and_entire_path_table_clearance():
    torch.manual_seed(47)
    env, command, _ = _commands_fixture(512)
    _realize_reset(env, command, torch.arange(512))
    right = command.angle_rad.cos() > 0
    assert .35 < float(right.float().mean()) < .65
    assert command.angle_rad[right].unique().numel() > 100
    assert (command.distance_m >= .2).all() and (command.distance_m <= .3).all()
    expected = torch.stack((-command.angle_rad.sin(), command.angle_rad.cos(), torch.zeros(512)), -1)
    torch.testing.assert_close(command.direction_w, expected, atol=1.e-6, rtol=0)
    padding = math.sqrt(3)*.03 + .01
    for t in torch.linspace(0, 1, 21):
        position = torch.lerp(command.target_pos_w, command.goal_pos_w, t)
        assert ((position[:, :2] - torch.tensor((-.75, 0.))).abs() <= torch.tensor((.18, .5))-padding).all()


def test_reset_snapshot_fixed_goal_and_partial_resets_preserve_other_rows():
    env, command, _ = _commands_fixture()
    env.scene.env_origins[1, 0] = 5.
    env.base_position[1, 0] = 5.
    command.set_reset_specs([1], [[-.69, .06, 1.08]], [math.pi], [.3])
    _realize_reset(env, command, torch.arange(4))
    initial, goal, angles = command.target_pos_w.clone(), command.goal_pos_w.clone(), command.angle_rad.clone()
    torch.testing.assert_close(initial[1], torch.tensor((4.31, .06, 1.08)))
    torch.testing.assert_close(goal[1], torch.tensor((4.31, -.24, 1.08)), atol=1.e-6, rtol=0)
    env.scene["target_object"].data.root_pos_w[:, 0] += .02
    command.compute(.02)
    torch.testing.assert_close(command.target_pos_w, initial)
    torch.testing.assert_close(command.goal_pos_w, goal)
    torch.testing.assert_close(command.angle_rad, angles)
    command.set_reset_specs([2], [-.70, 0., 1.08], 0., .25)
    _realize_reset(env, command, [2])
    torch.testing.assert_close(command.goal_pos_w[[0, 1, 3]], goal[[0, 1, 3]])


def test_invalid_fixture_rejected_before_buffers_or_scene_change():
    import pytest
    env, command, _ = _commands_fixture()
    before = env.scene["target_object"].data.root_pos_w.clone()
    for position, angle, distance in (([-.6, 0., 1.08], 0., .25), ([-.70, 0., 1.08], math.pi/2, .25),
                                      ([-.70, 0., 1.08], 0., .5), ([-.70, 0., 1.08], float("nan"), .25)):
        with pytest.raises(ValueError):
            command.set_reset_specs([0], position, angle, distance)
    assert not command._specified.any()
    torch.testing.assert_close(env.scene["target_object"].data.root_pos_w, before)


def test_current_cube_base_observation_is_unclipped_metres_and_command_encoding():
    env, command, observations = _commands_fixture()
    command.set_reset_specs([0], [-.70, 0., 1.08], math.pi/18, .3)
    _realize_reset(env, command, torch.arange(4))
    result = observations.push_command_observation(env)
    torch.testing.assert_close(result[0], torch.tensor((math.sin(math.pi/18), math.cos(math.pi/18), .3)))
    env.scene["target_object"].data.root_pos_w[0] = torch.tensor((-2.5, 3., 4.))
    result = observations.current_cube_base_position(env)
    torch.testing.assert_close(result[0], torch.tensor((2.5, -3., 4.-.79505)))
    from hand_manipulation_test.constants import PUSH_OBSERVATION_DIM, PUSH_OBSERVATION_SLICES
    assert PUSH_OBSERVATION_DIM == 61
    assert PUSH_OBSERVATION_SLICES["push_command"] == slice(55, 58)
    assert PUSH_OBSERVATION_SLICES["current_cube_base_position"] == slice(58, 61)


def test_command_cache_generation_subset_reset_preserves_other_rows_at_same_physics_counter():
    env, command, _ = _commands_fixture(2)
    command.set_reset_specs([0, 1], [-.70, 0., 1.08], 0., .25)
    _realize_reset(env, command, torch.arange(2))
    env.scene["target_object"].data.root_pos_w[:] = command.goal_pos_w
    env.scene["cube_palm_contacts"].data.force_matrix_w_history[:, :, 0, 0, 1] = .01
    env.scene["cube_table_contacts"].data.force_matrix_w_history[:, :, 0, 0, 2] = .01
    env.episode_length_buf[:] = 1
    env._sim_step_counter = 2
    # Termination queries precede rewards, and metrics query the same state
    # after both. Every query must reuse the one physical transition.
    before = command.state()
    assert before.first_contact.tolist() == [True, True]
    torch.testing.assert_close(before.settled_time_s, torch.full((2,), .02))
    assert command.state() is before
    command.compute(.02)
    assert command.state() is before

    command.set_reset_specs([1], [-.70, 0., 1.08], 0., .25)
    _realize_reset(env, command, [1])
    # Match the standard RL reset order: the command resamples before the
    # episode counter clears; stale contact history is masked afterwards.
    env.episode_length_buf[1] = 0
    after = command.state()
    assert after is not before
    assert command.state() is after
    assert after.first_contact.tolist() == [True, False]
    assert after.contact_seen.tolist() == [True, False]
    torch.testing.assert_close(after.settled_time_s, torch.tensor((.02, 0.)))
    torch.testing.assert_close(command.metrics["settled_time_s"], after.settled_time_s)
    torch.testing.assert_close(before.settled_time_s, torch.full((2,), .02))

    env._sim_step_counter = 4
    env.episode_length_buf[:] = 2
    next_step = command.state()
    assert next_step.first_contact.tolist() == [False, True]
    torch.testing.assert_close(next_step.settled_time_s, torch.tensor((.04, 0.)))
    assert command.state() is next_step
