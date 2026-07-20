"""Shared tensor helpers for the standalone sweep task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def pose_w_to_root_rpy(robot, position_w: torch.Tensor, quaternion_w: torch.Tensor) -> torch.Tensor:
    """Convert a world pose to robot-root ``xyz + roll/pitch/yaw``."""
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position_w,
        quaternion_w,
    )
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion_b)
    return torch.cat((position_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1)


def target_contact_data_w(
    env,
    sensor_names: tuple[str, ...] = ("left_contact", "right_contact"),
    force_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return force-weighted contact position, summed force, and contact mask."""
    point_numerator = torch.zeros(env.num_envs, 3, device=env.device)
    force_sum_w = torch.zeros_like(point_numerator)
    weight_sum = torch.zeros(env.num_envs, 1, device=env.device)

    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        force_matrix_w = sensor.data.force_matrix_w
        contact_pos_w = sensor.data.contact_pos_w
        if force_matrix_w is None or contact_pos_w is None:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' must track filtered forces and contact points."
            )
        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        safe_points = torch.nan_to_num(contact_pos_w, nan=0.0)
        weights = torch.linalg.norm(force_matrix_w, dim=-1) * valid.float()
        point_numerator += torch.sum(safe_points * weights.unsqueeze(-1), dim=(1, 2))
        force_sum_w += torch.sum(force_matrix_w * valid.unsqueeze(-1), dim=(1, 2))
        weight_sum += torch.sum(weights, dim=(1, 2)).unsqueeze(-1)

    contact_mask = weight_sum.squeeze(-1) > force_threshold
    contact_point_w = point_numerator / torch.clamp(weight_sum, min=1.0e-6)
    contact_point_w = torch.where(
        contact_mask.unsqueeze(-1), contact_point_w, torch.zeros_like(contact_point_w)
    )
    return contact_point_w, force_sum_w, contact_mask


def filtered_contact_mask(env, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Return whether a filtered contact sensor reports meaningful force."""
    sensor = env.scene[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' has no filtered force matrix.")
    force = torch.linalg.norm(force_matrix_w, dim=-1)
    return torch.any(force > force_threshold, dim=(1, 2))


def desired_direction_b(env, command_name: str) -> torch.Tensor:
    """Return the commanded planar direction in the robot base frame."""
    command = env.command_manager.get_command(command_name)
    return torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)


def object_displacement_b(env, command_name: str) -> torch.Tensor:
    """Return current object displacement from the command's initial pose."""
    term = env.command_manager.get_term(command_name)
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        term.robot.data.root_pos_w,
        term.robot.data.root_quat_w,
        term.target.data.root_pos_w,
    )
    return object_pos_b - term.initial_pose_b[:, :3]
