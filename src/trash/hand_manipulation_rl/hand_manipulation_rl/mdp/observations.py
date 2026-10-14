"""Exact 57-D actor/critic observation assembly."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

from ..assets.robot import ARM_JOINT_NAMES
from .episode_state import OBSERVATION_DIM, OBSERVATION_LAYOUT, OBSERVATION_SLICES


def _arm_joint_ids(env) -> list[int]:
    if not hasattr(env, "_observation_arm_joint_ids"):
        ids, names = env.scene["robot"].find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ARM_JOINT_NAMES:
            raise RuntimeError(f"Observation arm joint order mismatch: {names}")
        env._observation_arm_joint_ids = ids
    return env._observation_arm_joint_ids


def policy_observation(env) -> torch.Tensor:
    """Build the fixed-order policy vector without online object/contact GT.

    Object state is read only through the episode-fixed reset snapshot already
    expressed in C0.  Current object pose, true contact normal, board load,
    physical parameters and target identity do not enter this function.
    """

    robot = env.scene["robot"]
    arm_ids = _arm_joint_ids(env)
    observation = torch.zeros((env.num_envs, OBSERVATION_DIM), device=env.device)

    q = robot.data.joint_pos[:, arm_ids]
    dq = robot.data.joint_vel[:, arm_ids]
    observation[:, OBSERVATION_SLICES["arm_joint_position"]] = torch.clamp(q / math.pi, -1.5, 1.5)
    observation[:, OBSERVATION_SLICES["arm_joint_velocity"]] = torch.clamp(
        dq / env.cfg.task.arm_velocity_observation_scale, -1.0, 1.0
    )

    c_pos, c_quat = env.control_point_pose_w()
    relative_pos, relative_quat = math_utils.subtract_frame_transforms(
        env.c0_pos_w, env.c0_quat_w, c_pos, c_quat
    )
    relative_rotvec = math_utils.axis_angle_from_quat(math_utils.quat_unique(relative_quat))
    observation[:, OBSERVATION_SLICES["c_relative_position"]] = torch.clamp(
        relative_pos / env.cfg.task.position_observation_scale_m, -1.0, 1.0
    )
    observation[:, OBSERVATION_SLICES["c_relative_orientation"]] = torch.clamp(
        relative_rotvec / env.cfg.task.rotation_observation_scale_rad, -1.0, 1.0
    )

    hand_term = env.action_manager.get_term("hand_action")
    actual_hand = hand_term.actual_synergy
    if actual_hand.shape != (env.num_envs, 2):
        raise RuntimeError(f"Hand action term returned invalid actual state {tuple(actual_hand.shape)}")
    observation[:, OBSERVATION_SLICES["hand_state"]] = actual_hand

    sensor_live = env.sensor_valid & env.sensor_data_fresh
    tactile = env.selected_tactile_bits()
    # A just-reset row has an explicitly initialized, collision-certified
    # no-contact buffer.  Mask stale backend tensors to that valid zero value
    # until the first ordinary (action-bearing) physics substep refreshes them.
    tactile = torch.where(sensor_live.unsqueeze(-1), tactile, torch.zeros_like(tactile))
    header_and_tactile = torch.cat((env.surface_mode.float().unsqueeze(-1), tactile), dim=-1)
    observation[:, OBSERVATION_SLICES["surface_header_and_tactile"]] = header_and_tactile

    wrench = env.wrist_wrench_c().measured_c
    force = torch.clamp(
        wrench[:, :3] / env.cfg.task.wrench_force_observation_scale_n, -1.0, 1.0
    )
    moment = torch.clamp(
        wrench[:, 3:] / env.cfg.task.wrench_moment_observation_scale_nm, -1.0, 1.0
    )
    wrench_scaled = torch.cat((force, moment), dim=-1)
    wrench_scaled = torch.where(
        sensor_live.unsqueeze(-1), wrench_scaled, torch.zeros_like(wrench_scaled)
    )
    observation[:, OBSERVATION_SLICES["wrist_wrench_c"]] = wrench_scaled

    observation[:, OBSERVATION_SLICES["initial_object_relative_position"]] = torch.clamp(
        env.initial_object_pos_c0 / env.cfg.task.position_observation_scale_m, -1.0, 1.0
    )
    # The task has only shelf-left/shelf-right commands.  Encode their world-Y
    # sign directly and symmetrically as -1/+1 while preserving the 57-D layout.
    observation[:, OBSERVATION_SLICES["command_direction_y_s"]] = (
        env.command_direction_w[:, 1]
    ).unsqueeze(-1)
    observation[:, OBSERVATION_SLICES["command_distance"]] = (
        env.command_distance / env.cfg.task.distance_observation_scale_m
    ).unsqueeze(-1)
    observation[:, OBSERVATION_SLICES["last_action"]] = torch.clamp(
        env.last_policy_action, -1.0, 1.0
    )

    if observation.shape[-1] != 57 or not torch.isfinite(observation).all():
        raise RuntimeError(f"Invalid policy observation shape/data: {tuple(observation.shape)}")
    return observation


def observation_contract() -> tuple[tuple[str, int], ...]:
    """Return the immutable human-readable term layout for probes/tests."""

    return OBSERVATION_LAYOUT


__all__ = ["OBSERVATION_DIM", "OBSERVATION_LAYOUT", "OBSERVATION_SLICES", "policy_observation"]
