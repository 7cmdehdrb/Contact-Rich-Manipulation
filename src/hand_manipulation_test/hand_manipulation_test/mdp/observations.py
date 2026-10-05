"""Measured robot state and reaching target observations."""

from __future__ import annotations

import math

import torch

from ..action_math import quaternion_conjugate, quaternion_multiply, quaternion_rotate
from ..geometry import base_link_pose_w, control_point_pose_w, palm_tactile_bits, wrist_wrench_c


def _target(env):
    return env.command_manager.get_term("target_position")


def _sensor_live(env) -> torch.Tensor:
    # Reset rows mask backend contact tensors from the last episode. The
    # getattr also supports ManagerBasedEnv probes and small test adapters.
    length = getattr(env, "episode_length_buf", None)
    if length is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return length > 0


def arm_joint_position(env) -> torch.Tensor:
    ids = env.action_manager.get_term("arm_action").joint_ids
    return torch.clamp(env.scene["robot"].data.joint_pos[:, ids] / math.pi, -1.5, 1.5)


def arm_joint_velocity(env) -> torch.Tensor:
    ids = env.action_manager.get_term("arm_action").joint_ids
    return torch.clamp(
        env.scene["robot"].data.joint_vel[:, ids] / env.cfg.task.arm_velocity_observation_scale,
        -1.0,
        1.0,
    )


def eef_relative_position(env) -> torch.Tensor:
    """Current C displacement from reset C0, expressed in C0, in metres."""

    target = _target(env)
    pos_w, _ = control_point_pose_w(env)
    return quaternion_rotate(
        quaternion_conjugate(target.initial_eef_quat_w),
        pos_w - target.initial_eef_pos_w,
    )


def eef_relative_orientation(env) -> torch.Tensor:
    """Shortest C0-to-C rotation vector, normalized by the rotation scale."""

    _, quat_w = control_point_pose_w(env)
    relative = quaternion_multiply(quaternion_conjugate(_target(env).initial_eef_quat_w), quat_w)
    relative = relative / torch.clamp_min(torch.linalg.vector_norm(relative, dim=-1, keepdim=True), 1.0e-12)
    relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
    vector = relative[:, 1:]
    sine_half_angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine_half_angle, relative[:, :1])
    scale = torch.where(
        sine_half_angle > 1.0e-7,
        angle / torch.clamp_min(sine_half_angle, 1.0e-12),
        torch.full_like(sine_half_angle, 2.0),
    )
    return torch.clamp(vector * scale / env.cfg.task.rotation_observation_scale_rad, -1.0, 1.0)


def hand_state(env) -> torch.Tensor:
    return env.action_manager.get_term("hand_action").actual_synergy


def surface_header_and_tactile(env) -> torch.Tensor:
    """Constant palm-mode header (0), followed by the 17 physical palm bits."""

    live = _sensor_live(env)
    bits = palm_tactile_bits(env).to(dtype=torch.float32)
    bits = torch.where(live.unsqueeze(-1), bits, torch.zeros_like(bits))
    header = torch.zeros((env.num_envs, 1), dtype=bits.dtype, device=bits.device)
    return torch.cat((header, bits), dim=-1)


def wrist_wrench(env) -> torch.Tensor:
    wrench = wrist_wrench_c(env).measured_c
    force = torch.clamp(wrench[:, :3] / env.cfg.task.wrench_force_observation_scale_n, -1.0, 1.0)
    moment = torch.clamp(wrench[:, 3:] / env.cfg.task.wrench_moment_observation_scale_nm, -1.0, 1.0)
    scaled = torch.cat((force, moment), dim=-1)
    return torch.where(_sensor_live(env).unsqueeze(-1), scaled, torch.zeros_like(scaled))


def initial_target_relative_position(env) -> torch.Tensor:
    """Episode-fixed target position in the robot's base_link frame, in metres.

    Subtract the actual base_link origin and express the vector in its axes.
    Moving or rotating the EEF does not change this position observation.
    """

    pos_w, quat_w = base_link_pose_w(env)
    return quaternion_rotate(quaternion_conjugate(quat_w), _target(env).target_pos_w - pos_w)


def last_action(env) -> torch.Tensor:
    # ActionManager.action retains caller inputs; each action term stores its
    # finite, bounded normalized input after sanitation.
    arm = env.action_manager.get_term("arm_action").raw_actions
    hand = env.action_manager.get_term("hand_action").raw_actions
    action = torch.cat((arm, hand), dim=-1)
    # A reset has no previous policy action. Hand targets are synchronized
    # from actual joints and their inverse mapping can have float roundoff.
    return torch.where(_sensor_live(env).unsqueeze(-1), action, torch.zeros_like(action))


def push_command_observation(env) -> torch.Tensor:
    """Episode-fixed [sin(theta), cos(theta), distance in metres]."""

    return _target(env).command


def current_cube_base_position(env) -> torch.Tensor:
    """Current physical Cube centre in actual base_link axes, in metres."""

    position, quaternion = base_link_pose_w(env)
    return quaternion_rotate(
        quaternion_conjugate(quaternion), env.scene["target_object"].data.root_pos_w - position
    )


__all__ = [
    "arm_joint_position", "arm_joint_velocity", "eef_relative_position", "eef_relative_orientation",
    "hand_state", "surface_header_and_tactile", "wrist_wrench", "initial_target_relative_position", "last_action",
    "push_command_observation", "current_cube_base_position",
]
