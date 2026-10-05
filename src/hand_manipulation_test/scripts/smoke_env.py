#!/usr/bin/env python3
"""Verify the reaching task against live Isaac Sim physics and manager contracts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
TASK_ID = "Isaac-Hand-Manipulation-Test-v0"
CONTACT_TASK_ID = "Isaac-Hand-Manipulation-Contact-v0"
PUSH_TASK_ID = "Isaac-Hand-Manipulation-Push-v0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(TASK_ID, CONTACT_TASK_ID, PUSH_TASK_ID), default=TASK_ID)
parser.add_argument("--num_envs", "--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=50, help="Hold steps (arm zero; contact hand holds initial openness).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--check-timeout", action="store_true", help="Run a complete 500-step episode.")
parser.add_argument("--debug-vis", action="store_true", help="Create the Target and EEF markers.")
parser.add_argument("--disable-markers", action="store_true")
parser.add_argument("--reset-samples", type=int, default=4, help="Additional contact-task safe resets to check.")
AppLauncher.add_app_launcher_args(parser)
args, extra_args = parser.parse_known_args()
if args.task == PUSH_TASK_ID:
    push_smoke = PACKAGE_ROOT / "scripts" / "smoke_push_env.py"
    os.execv(sys.executable, [sys.executable, str(push_smoke), *sys.argv[1:]])
if extra_args:
    parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
app_launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

import hand_manipulation_test  # noqa: E402, F401 -- registers the Gym task
from hand_manipulation_test.action_math import inspire_synergy_to_joint_positions  # noqa: E402
from hand_manipulation_test.assets.robot import (  # noqa: E402
    ARM_JOINT_NAMES,
    HAND_CLOSED_TARGETS,
    HAND_JOINT_NAMES,
    HAND_OPEN_TARGETS,
    ROBOT_CONTACT_BODY_NAMES,
)
from hand_manipulation_test.constants import OBSERVATION_SLICES  # noqa: E402
from hand_manipulation_test.geometry import (  # noqa: E402
    control_point_pose_w,
    palm_tactile_bits,
    wrist_wrench_c,
)
from hand_manipulation_test.sensors import PALM_CHANNEL_NAMES  # noqa: E402


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} contains non-finite data")


def _part(policy: torch.Tensor, name: str) -> torch.Tensor:
    return policy[:, OBSERVATION_SLICES[name]]


def _is_contact() -> bool:
    return args.task == CONTACT_TASK_ID


def _hold_action(env) -> torch.Tensor:
    action = torch.zeros((env.num_envs, 8), device=env.device)
    if _is_contact():
        action[:, 6:] = 2.0 * env.action_manager.get_term("hand_action").actual_synergy - 1.0
    return action


def _safe_term(env):
    return env.event_manager.get_term_cfg("safe_hand").func


def _reset_instrumentation(env) -> dict[str, torch.Tensor]:
    if not _is_contact():
        return {}
    term = _safe_term(env)
    return {name: getattr(term, name).clone() for name in ("attempts", "accepted_seed_index", "ik_iterations", "obb_checks")}


def _assert_reset_instrumentation(env, before) -> None:
    if before:
        after = _reset_instrumentation(env)
        for name, value in before.items():
            torch.testing.assert_close(after[name], value, rtol=0.0, atol=0.0)


def _osc_tracking_error(env, position_w, quaternion_w) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare fresh post-physics C data with the target latched for this policy step."""
    robot = env.scene["robot"]
    root_quaternion = robot.data.root_link_quat_w
    position_b = quat_apply_inverse(root_quaternion, position_w - robot.data.root_link_pos_w)
    quaternion_b = quat_mul(quat_conjugate(root_quaternion), quaternion_w)
    desired = env.action_manager.get_term("arm_action").desired_c_pose_b
    _finite("OSC target pose", desired)
    position_error = torch.linalg.vector_norm(position_b - desired[:, :3], dim=-1)
    cosine = torch.abs((quaternion_b * desired[:, 3:]).sum(dim=-1)).clamp(0.0, 1.0)
    return position_error, 2.0 * torch.acos(cosine)


def _check_policy(env, observation, *, reset_ids=None) -> None:
    policy = observation["policy"]
    assert policy.shape == (env.num_envs, 55), policy.shape
    _finite("policy observation", policy)
    tactile = _part(policy, "surface_header_and_tactile")
    torch.testing.assert_close(tactile[:, 0], torch.zeros_like(tactile[:, 0]), rtol=0.0, atol=0.0)
    if reset_ids is not None:
        for name in ("surface_header_and_tactile", "wrist_wrench_c", "last_action"):
            value = _part(policy, name)[reset_ids]
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(tactile[:, 1:], palm_tactile_bits(env), rtol=0.0, atol=0.0)
        wrench = wrist_wrench_c(env).measured_c
        expected_wrench = torch.cat((
            (wrench[:, :3] / env.cfg.task.wrench_force_observation_scale_n).clamp(-1.0, 1.0),
            (wrench[:, 3:] / env.cfg.task.wrench_moment_observation_scale_nm).clamp(-1.0, 1.0),
        ), dim=-1)
        torch.testing.assert_close(_part(policy, "wrist_wrench_c"), expected_wrench, rtol=0.0, atol=1.0e-6)
    position, quaternion = control_point_pose_w(env)
    frame = env.scene["ee_frame"].data
    assert frame.target_frame_names == ["end_effector"], frame.target_frame_names
    torch.testing.assert_close(frame.target_pos_w[:, 0], position, rtol=1.0e-5, atol=2.0e-6)
    same_orientation = torch.abs((frame.target_quat_w[:, 0] * quaternion).sum(dim=-1))
    torch.testing.assert_close(same_orientation, torch.ones_like(same_orientation), rtol=0.0, atol=2.0e-6)
    base_ids, base_names = env.scene["robot"].find_bodies("base_link", preserve_order=True)
    assert base_names == ["base_link"], base_names
    robot = env.scene["robot"]
    torch.testing.assert_close(
        frame.source_pos_w, robot.data.body_link_pos_w[:, base_ids[0]], rtol=0.0, atol=2.0e-6
    )
    base_orientation = torch.abs((frame.source_quat_w * robot.data.body_link_quat_w[:, base_ids[0]]).sum(dim=-1))
    torch.testing.assert_close(base_orientation, torch.ones_like(base_orientation), rtol=0.0, atol=2.0e-6)
    command = env.command_manager.get_term("target_position")
    relative = quat_apply_inverse(frame.source_quat_w, command.target_pos_w - frame.source_pos_w)
    torch.testing.assert_close(
        _part(policy, "initial_target_relative_position"), relative, rtol=1.0e-5, atol=2.0e-6
    )
    initial_relative = quat_apply_inverse(command.initial_eef_quat_w, position - command.initial_eef_pos_w)
    torch.testing.assert_close(
        _part(policy, "eef_relative_position"), initial_relative, rtol=1.0e-5, atol=2.0e-6
    )


def _check_initial_state(env, observation, env_ids) -> None:
    robot = env.scene["robot"]
    arm_ids, names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    assert tuple(names) == tuple(ARM_JOINT_NAMES)
    if not _is_contact():
        expected_arm = robot.data.joint_pos.new_tensor((0.0, -2.2, 2.2, 0.0, 1.57, 0.785))
        torch.testing.assert_close(
            robot.data.joint_pos[env_ids][:, arm_ids], expected_arm.expand(len(env_ids), -1),
            rtol=0.0, atol=1.0e-6,
        )
    else:
        actual_arm = robot.data.joint_pos[env_ids][:, arm_ids]
        limits = robot.data.soft_joint_pos_limits[env_ids][:, arm_ids]
        assert bool(((actual_arm >= limits[..., 0]) & (actual_arm <= limits[..., 1])).all())
    torch.testing.assert_close(
        robot.data.joint_vel[env_ids], torch.zeros_like(robot.data.joint_vel[env_ids]), rtol=0.0, atol=1.0e-6
    )
    hand_ids, names = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    assert tuple(names) == tuple(HAND_JOINT_NAMES)
    synergy = _part(observation["policy"], "hand_state")[env_ids]
    if _is_contact():
        low, high = env.cfg.task.initial_hand_open_range
        assert bool(((synergy >= low - 1.0e-6) & (synergy <= high + 1.0e-6)).all()), synergy.tolist()
    else:
        torch.testing.assert_close(synergy, torch.full_like(synergy, 0.5), rtol=0.0, atol=1.0e-6)
    open_hand = robot.data.joint_pos.new_tensor([HAND_OPEN_TARGETS[name] for name in HAND_JOINT_NAMES])
    closed_hand = robot.data.joint_pos.new_tensor([HAND_CLOSED_TARGETS[name] for name in HAND_JOINT_NAMES])
    expected_hand = inspire_synergy_to_joint_positions(synergy, open_hand, closed_hand)
    torch.testing.assert_close(
        robot.data.joint_pos[env_ids][:, hand_ids], expected_hand,
        rtol=0.0, atol=1.0e-6,
    )
    for name in ("eef_relative_position", "eef_relative_orientation"):
        value = _part(observation["policy"], name)[env_ids]
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0.0, atol=2.0e-6)
    _check_policy(env, observation, reset_ids=env_ids)
    if _is_contact():
        term = _safe_term(env)
        assert bool(term.check_final(env_ids).all()), "Reset failed final pose/clearance certification"
        assert bool((term.attempts[env_ids] > 0).all())
        assert bool((term.accepted_seed_index[env_ids] >= 0).all())
        palm_position, hand_quaternion = term.palm_reference_pose_w(env_ids)
        palm_normal = quat_apply(hand_quaternion, hand_quaternion.new_tensor((0.0, 1.0, 0.0)).expand(len(env_ids), -1))
        target = env.command_manager.get_term("target_position").target_pos_w[env_ids]
        direction = target - palm_position
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        assert bool(((palm_normal * direction).sum(dim=-1) >= torch.cos(direction.new_tensor(0.05)) - 1.0e-6).all())
        finger = quat_apply(hand_quaternion, hand_quaternion.new_tensor((0.0, 0.0, 1.0)).expand(len(env_ids), -1))
        outward = env.scene["table"].data.root_pos_w[env_ids] - robot.data.root_link_pos_w[env_ids]
        assert bool(((finger[:, :2] * outward[:, :2]).sum(dim=-1) > 0.0).all()), "Fingers point toward the actual robot base"
        table_local = env.scene["table"].data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        torch.testing.assert_close(table_local, table_local.new_tensor(env.cfg.task.table_center).expand(len(env_ids), -1), rtol=0.0, atol=5.0e-6)
        torch.testing.assert_close(env.scene["target_object"].data.root_pos_w[env_ids], target, rtol=0.0, atol=2.0e-6)
        torch.testing.assert_close(
            env.scene["target_object"].data.root_vel_w[env_ids],
            torch.zeros_like(env.scene["target_object"].data.root_vel_w[env_ids]), rtol=0.0, atol=1.0e-6,
        )


def _check_target_range(env) -> None:
    command = env.command_manager.get_term("target_position")
    local = command.target_pos_w - env.scene.env_origins
    tolerance = 2.0e-6
    if _is_contact():
        low, high = env.cfg.task.object_xy_range_low, env.cfg.task.object_xy_range_high
        height = env.cfg.task.cube_center_height_m
    else:
        low, high = command.cfg.target_position_range_low[:2], command.cfg.target_position_range_high[:2]
        height = command.cfg.target_position_range_low[2]
    for axis in range(2):
        assert bool(((local[:, axis] >= low[axis] - tolerance) & (local[:, axis] <= high[axis] + tolerance)).all())
    torch.testing.assert_close(local[:, 2], torch.full_like(local[:, 2], height), rtol=0.0, atol=tolerance)


def _step(env, action):
    observation, reward, terminated, truncated, extras = env.step(action)
    if bool((terminated | truncated).any()):
        raise AssertionError(f"Unexpected episode end: terminated={terminated.tolist()}, truncated={truncated.tolist()}")
    _check_policy(env, observation)
    _finite("reward", reward)
    _finite("joint positions", env.scene["robot"].data.joint_pos)
    _finite("joint velocities", env.scene["robot"].data.joint_vel)
    if _is_contact():
        matrix = env.scene["cube_palm_contacts"].data.force_matrix_w
        history = env.scene["table_contacts"].data.force_matrix_w_history
        _finite("Cube-palm force matrix", matrix)
        _finite("Robot-table force history", history)
        assert matrix.shape == (env.num_envs, 1, 17, 3), matrix.shape
        assert history.shape == (env.num_envs, 2, 1, len(ROBOT_CONTACT_BODY_NAMES), 3), history.shape
        expected_contact = (torch.linalg.vector_norm(matrix, dim=-1).flatten(1).amax(-1) >= env.cfg.task.contact_threshold_n).float()
        position, _ = control_point_pose_w(env)
        target = env.command_manager.get_term("target_position").target_pos_w
        expected_distance = 4.0 * torch.exp(-torch.linalg.vector_norm(position - target, dim=-1) / env.cfg.task.goal_reward_sigma_m)
        for env_index in range(env.num_envs):
            terms = {name: values[0] for name, values in env.reward_manager.get_active_iterable_terms(env_index)}
            torch.testing.assert_close(reward[env_index], reward.new_tensor(sum(terms.values()) * env.step_dt), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(reward.new_tensor(terms["eef_distance"]), expected_distance[env_index], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(reward.new_tensor(terms["palm_cube_contact"]), 0.02 * expected_contact[env_index], atol=1e-6, rtol=0.0)
    return observation, reward, terminated, truncated, extras


def _check_scene_and_markers(env, command) -> None:
    for prim in env.sim.stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        if path.startswith("/World/envs/"):
            if not _is_contact():
                assert name not in {"cube", "board", "shelf", "target_object"}, path
            assert "dorsal" not in name, path
        if path.startswith("/Visuals/HandManipulationTest/"):
            assert not prim.HasAPI(UsdPhysics.CollisionAPI), path
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI), path
    if env.cfg.commands.target_position.debug_vis:
        assert command._target_marker is not None
        command._debug_vis_callback(None)
        assert command._target_marker.count == env.num_envs
    if env.cfg.scene.ee_frame.debug_vis:
        frame = env.scene["ee_frame"]
        assert frame.data.target_frame_names == ["end_effector"]
        assert hasattr(frame, "frame_visualizer")
        frame._debug_vis_callback(None)
        # Native FrameTransformer draws source and target frames plus one
        # connecting line for each environment in a single point instancer.
        assert frame.frame_visualizer.count == 3 * env.num_envs


def _check_contact_scene(env) -> None:
    assert set(env.scene.rigid_objects) == {"table", "target_object"}
    assert set(env.scene.sensors) == {"palm_tactile", "ee_frame", "table_contacts", "cube_palm_contacts"}
    assert env.cfg.scene.table.spawn.rigid_props.kinematic_enabled is True
    assert env.cfg.scene.target_object.spawn.rigid_props.kinematic_enabled is False
    assert env.cfg.scene.target_object.spawn.rigid_props.disable_gravity is False
    assert env.cfg.scene.table_contacts.history_length == 2
    assert len(env.cfg.scene.table_contacts.filter_prim_paths_expr) == len(ROBOT_CONTACT_BODY_NAMES) == 38
    assert len(env.cfg.scene.cube_palm_contacts.filter_prim_paths_expr) == 17
    assert not any("TargetObject" in path for path in env.cfg.scene.table_contacts.filter_prim_paths_expr)
    weights = {name: env.reward_manager.get_term_cfg(name).weight for name in env.reward_manager.active_terms}
    assert weights == {"eef_distance": 4.0, "action_rate": 0.001, "palm_cube_contact": 0.02}, weights


def _check_contact_resets(env) -> list[float]:
    """Check every box corner and bounded random resets without advancing physics."""
    ids = torch.arange(env.num_envs, device=env.device)
    task = env.cfg.task
    original_low, original_high = task.object_xy_range_low, task.object_xy_range_high
    original_offset_low, original_offset_high = task.object_xy_offset_low, task.object_xy_offset_high
    durations = []
    try:
        for x in (original_low[0], original_high[0]):
            for y in (original_low[1], original_high[1]):
                task.object_xy_offset_low = task.object_xy_offset_high = (x - task.table_center[0], y - task.table_center[1])
                counter = env._sim_step_counter
                started = time.perf_counter()
                observation, _ = env.reset()
                durations.append(time.perf_counter() - started)
                assert env._sim_step_counter == counter, "Safe reset must not advance the physics clock"
                _check_initial_state(env, observation, ids)
                _check_target_range(env)
                local = env.scene["target_object"].data.root_pos_w - env.scene.env_origins
                torch.testing.assert_close(local[:, :2], local.new_tensor((x, y)).expand(env.num_envs, -1), atol=2e-6, rtol=0.0)
    finally:
        task.object_xy_offset_low, task.object_xy_offset_high = original_offset_low, original_offset_high
    for _ in range(args.reset_samples):
        counter = env._sim_step_counter
        started = time.perf_counter()
        observation, _ = env.reset()
        durations.append(time.perf_counter() - started)
        assert env._sim_step_counter == counter
        _check_initial_state(env, observation, ids)
        _check_target_range(env)
    return durations


def _check_movable_cube_fixed_target(env, observation):
    """Physically move the Cube and retain the snapshot observation and reward goal."""
    cube = env.scene["target_object"]
    command = env.command_manager.get_term("target_position")
    fixed_target = command.target_pos_w.clone()
    fixed_observation = _part(observation["policy"], "initial_target_relative_position").clone()
    distance_func = env.reward_manager.get_term_cfg("eef_distance").func
    before_reward = distance_func(env).clone()
    pose = cube.data.root_pose_w.clone()
    position, _ = control_point_pose_w(env)
    sign = torch.where(pose[:, 1] >= position[:, 1], 1.0, -1.0)
    pose[:, 1] += 0.08 * sign
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
    torch.testing.assert_close(distance_func(env), before_reward, rtol=0.0, atol=0.0)
    observation, _, _, _, _ = _step(env, _hold_action(env))
    torch.testing.assert_close(command.target_pos_w, fixed_target, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_part(observation["policy"], "initial_target_relative_position"), fixed_observation, rtol=0.0, atol=2e-6)
    assert bool((torch.linalg.vector_norm(cube.data.root_pos_w - fixed_target, dim=-1) > 0.06).all())
    return observation


def main() -> None:
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 50 or args.steps >= 500:
        raise ValueError("--steps must be in [50, 499]; use --check-timeout for a full episode")
    if args.reset_samples < 0:
        raise ValueError("--reset-samples cannot be negative")
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.commands.target_position.debug_vis = args.debug_vis and not args.disable_markers
    cfg.scene.ee_frame.debug_vis = args.debug_vis and not args.disable_markers
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        all_ids = torch.arange(env.num_envs, device=env.device)
        started = time.perf_counter()
        observation, _ = env.reset(seed=args.seed)
        reset_durations = [time.perf_counter() - started]
        assert env.action_manager.total_action_dim == 8
        expected_rewards = {"eef_distance", "action_rate", "palm_cube_contact"} if _is_contact() else {"eef_distance", "action_rate"}
        expected_terminations = {"time_out", "table_contact"} if _is_contact() else {"time_out"}
        assert set(env.reward_manager.active_terms) == expected_rewards
        assert set(env.termination_manager.active_terms) == expected_terminations
        assert env.termination_manager.get_term_cfg("time_out").time_out is True
        assert env.max_episode_length == 500
        assert abs(env.step_dt - 0.02) < 1.0e-9
        assert cfg.actions.arm_action.gravity_compensation is True
        if _is_contact():
            _check_contact_scene(env)
        else:
            assert set(env.scene.sensors) == {"palm_tactile", "ee_frame"}
            assert not env.scene.rigid_objects, list(env.scene.rigid_objects)
        assert set(env.scene["palm_tactile"].body_names) == set(PALM_CHANNEL_NAMES)
        assert len(env.scene["palm_tactile"].body_names) == 17
        assert env.scene["robot"].root_physx_view is not None
        _check_initial_state(env, observation, all_ids)
        _check_target_range(env)
        if _is_contact():
            reset_durations.extend(_check_contact_resets(env))
            observation, _ = env.reset()
            _check_initial_state(env, observation, all_ids)
        command = env.command_manager.get_term("target_position")
        _check_scene_and_markers(env, command)
        if env.num_envs > 1:
            assert bool((env.scene.env_origins[0] != env.scene.env_origins[1]).any())
            if not _is_contact():
                initial_local = command.initial_eef_pos_w - env.scene.env_origins
                torch.testing.assert_close(initial_local, initial_local[:1].expand_as(initial_local), atol=2.0e-5, rtol=0.0)
        target_before = command.target_pos_w.clone()
        initial_target_observation = _part(observation["policy"], "initial_target_relative_position").clone()
        reset_position, reset_quaternion = control_point_pose_w(env)
        reset_position, reset_quaternion = reset_position.clone(), reset_quaternion.clone()
        action = _hold_action(env)
        instrumentation_before = _reset_instrumentation(env)

        # A nonzero first action is unpenalized; subsequent changes use executed actions.
        action[:, 0] = 0.2
        action[:, 5] = 0.2
        observation, _, terminated, truncated, _ = _step(env, action)
        assert not bool((terminated | truncated).any())
        torch.testing.assert_close(
            _part(observation["policy"], "initial_target_relative_position"), initial_target_observation,
            rtol=0.0, atol=2.0e-6,
        )
        rate_index = env.reward_manager.active_terms.index("action_rate")
        for env_index in range(env.num_envs):
            rate = env.reward_manager.get_active_iterable_terms(env_index)[rate_index][1][0]
            assert abs(rate) < 1.0e-9, rate
        action[:, 0] = -0.2
        for _ in range(7):
            observation, _, terminated, truncated, _ = _step(env, action)
            assert not bool((terminated | truncated).any())
            torch.testing.assert_close(command.target_pos_w, target_before, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                _part(observation["policy"], "initial_target_relative_position"), initial_target_observation,
                rtol=0.0, atol=2.0e-6,
            )
        current_position, current_quaternion = control_point_pose_w(env)
        assert bool((torch.linalg.vector_norm(current_position - reset_position, dim=-1) > 1.0e-5).all())
        assert bool((torch.abs(current_quaternion - reset_quaternion).amax(dim=-1) > 1.0e-5).all())
        _assert_reset_instrumentation(env, instrumentation_before)
        if _is_contact():
            observation = _check_movable_cube_fixed_target(env, observation)
            _assert_reset_instrumentation(env, instrumentation_before)

        # Reset only env zero: other environments retain target, pose and action history.
        if env.num_envs > 1:
            untouched_target = command.target_pos_w[1:].clone()
            untouched_target_observation = _part(observation["policy"], "initial_target_relative_position")[1:].clone()
            untouched_joints = env.scene["robot"].data.joint_pos[1:].clone()
            untouched_action = _part(observation["policy"], "last_action")[1:].clone()
            if _is_contact():
                untouched_cube_pose = env.scene["target_object"].data.root_pose_w[1:].clone()
                untouched_cube_velocity = env.scene["target_object"].data.root_vel_w[1:].clone()
                untouched_instrumentation = _reset_instrumentation(env)
            ids = all_ids[:1]
            observation, _ = env.reset(env_ids=ids)
            _check_initial_state(env, observation, ids)
            _check_target_range(env)
            torch.testing.assert_close(command.target_pos_w[1:], untouched_target, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                _part(observation["policy"], "initial_target_relative_position")[1:], untouched_target_observation,
                rtol=0.0, atol=2.0e-6,
            )
            torch.testing.assert_close(env.scene["robot"].data.joint_pos[1:], untouched_joints, rtol=0.0, atol=0.0)
            torch.testing.assert_close(_part(observation["policy"], "last_action")[1:], untouched_action, rtol=0.0, atol=0.0)
            if _is_contact():
                torch.testing.assert_close(env.scene["target_object"].data.root_pose_w[1:], untouched_cube_pose, rtol=0.0, atol=0.0)
                torch.testing.assert_close(env.scene["target_object"].data.root_vel_w[1:], untouched_cube_velocity, rtol=0.0, atol=0.0)
                for name, value in _reset_instrumentation(env).items():
                    torch.testing.assert_close(value[1:], untouched_instrumentation[name][1:], rtol=0.0, atol=0.0)
            observation, _, _, _, _ = _step(env, _hold_action(env))
            zero_rate = env.reward_manager.get_active_iterable_terms(0)[rate_index][1][0]
            assert abs(zero_rate) < 1.0e-9, zero_rate

        observation, _ = env.reset()
        _check_initial_state(env, observation, all_ids)
        initial_position, initial_quaternion = control_point_pose_w(env)
        initial_position, initial_quaternion = initial_position.clone(), initial_quaternion.clone()
        fixed_target = command.target_pos_w.clone()
        fixed_target_observation = _part(observation["policy"], "initial_target_relative_position").clone()
        action = _hold_action(env)
        instrumentation_before = _reset_instrumentation(env)
        peak_drift = torch.zeros(env.num_envs, device=env.device)
        peak_angle = torch.zeros_like(peak_drift)
        peak_tracking_position_error = torch.zeros_like(peak_drift)
        peak_tracking_angle_error = torch.zeros_like(peak_drift)
        force_peak = torch.zeros_like(peak_drift)
        normal_step_started = time.perf_counter()
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = _step(env, action)
            assert not bool((terminated | truncated).any())
            torch.testing.assert_close(command.target_pos_w, fixed_target, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                _part(observation["policy"], "initial_target_relative_position"), fixed_target_observation,
                rtol=0.0, atol=2.0e-6,
            )
            position, quaternion = control_point_pose_w(env)
            peak_drift = torch.maximum(peak_drift, torch.linalg.vector_norm(position - initial_position, dim=-1))
            cosine = torch.abs((quaternion * initial_quaternion).sum(dim=-1)).clamp(0.0, 1.0)
            peak_angle = torch.maximum(peak_angle, 2.0 * torch.acos(cosine))
            tracking_position_error, tracking_angle_error = _osc_tracking_error(env, position, quaternion)
            peak_tracking_position_error = torch.maximum(peak_tracking_position_error, tracking_position_error)
            peak_tracking_angle_error = torch.maximum(peak_tracking_angle_error, tracking_angle_error)
            sample = wrist_wrench_c(env)
            torch.testing.assert_close(sample.measured_f, sample.raw_f, rtol=0.0, atol=0.0)
            _finite("raw wrist force/torque", sample.raw_f)
            force_peak = torch.maximum(force_peak, torch.linalg.vector_norm(sample.raw_f[:, :3], dim=-1))
        normal_step_seconds = time.perf_counter() - normal_step_started
        _assert_reset_instrumentation(env, instrumentation_before)
        if _is_contact():
            # Zero Cartesian deltas re-anchor the target to the current C pose
            # each policy step; measure tracking against that actual command.
            assert bool((peak_tracking_position_error < 0.02).all()), peak_tracking_position_error.tolist()
            assert bool((peak_tracking_angle_error < 0.05).all()), peak_tracking_angle_error.tolist()
        else:
            assert bool((peak_drift < 0.02).all()), peak_drift.tolist()
            assert bool((peak_angle < 0.05).all()), peak_angle.tolist()
        assert bool((force_peak > 0.01).all()), "Live F/T must retain the hand's gravitational load"
        if args.check_timeout:
            observation, _ = env.reset()
            action = _hold_action(env)
            fixed_target = command.target_pos_w.clone()
            for step_index in range(500):
                observation, reward, terminated, truncated, _ = env.step(action)
                _finite("timeout observation", observation["policy"])
                _finite("timeout reward", reward)
                assert not bool(terminated.any())
                if step_index < 499:
                    assert not bool(truncated.any()), step_index
                    torch.testing.assert_close(command.target_pos_w, fixed_target, rtol=0.0, atol=0.0)
                else:
                    assert bool(truncated.all()), "Every environment must truncate at step 500"
                    _check_initial_state(env, observation, all_ids)
        print(
            "[PASS] 55D observations, 8D actions, initial joints/hand, palm/F/T sensors, "
            "base-link target frame, current-EEF control/markers, partial resets, fixed targets and holding action",
            {"peak_position_drift_m": peak_drift.tolist(), "peak_angle_drift_rad": peak_angle.tolist(),
             "peak_tracking_position_error_m": peak_tracking_position_error.tolist(),
             "peak_tracking_angle_error_rad": peak_tracking_angle_error.tolist(),
             "peak_raw_ft_force_n": force_peak.tolist(), "timeout_checked": args.check_timeout},
            {"task": args.task, "reset_seconds": reset_durations, "hold_step_seconds_mean": normal_step_seconds / args.steps,
             "safe_reset_counters": {name: value.tolist() for name, value in _reset_instrumentation(env).items()}},
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
