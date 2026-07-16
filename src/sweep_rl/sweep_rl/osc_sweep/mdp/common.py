"""Shared tensor helpers for sweep observations, rewards, and terminations."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor


def pose_w_to_root_rpy(
    robot: Articulation,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """Convert a world pose to robot-root ``xyz + roll/pitch/yaw``."""
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        pos_w,
        quat_w,
    )
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_b)
    return torch.cat(
        (pos_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
    )


def target_contact_data_w(
    env,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
    force_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return force-weighted contact point, summed normal force, and contact mask.

    Contact sensors are filtered against the target cube. A zero point is
    returned when neither gripper pad is in contact.
    """
    point_numerator = torch.zeros(env.num_envs, 3, device=env.device)
    force_sum_w = torch.zeros_like(point_numerator)
    weight_sum = torch.zeros(env.num_envs, 1, device=env.device)

    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        force_matrix_w = sensor.data.force_matrix_w
        contact_pos_w = sensor.data.contact_pos_w
        if force_matrix_w is None or contact_pos_w is None:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' must track filtered forces "
                "and contact points."
            )

        force_sum_w += force_matrix_w.sum(dim=(1, 2))
        weights = torch.linalg.norm(force_matrix_w, dim=-1)
        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        weights = torch.where(valid, weights, 0.0)
        safe_points = torch.nan_to_num(contact_pos_w, nan=0.0)
        point_numerator += (safe_points * weights.unsqueeze(-1)).sum(
            dim=(1, 2)
        )
        weight_sum += weights.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)

    contact_mask = weight_sum.squeeze(-1) > force_threshold
    contact_point_w = point_numerator / torch.clamp(weight_sum, min=1.0e-6)
    contact_point_w = torch.where(
        contact_mask.unsqueeze(-1),
        contact_point_w,
        torch.zeros_like(contact_point_w),
    )
    return contact_point_w, force_sum_w, contact_mask


def object_displacement_b(env, command_name: str) -> torch.Tensor:
    """Current target-object displacement from the command's initial pose."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.robot_name]
    target: RigidObject = env.scene[command.cfg.object_name]
    current_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
    )
    return current_pos_b - command.initial_pose_b[:, :3]


def desired_direction_b(env, command_name: str) -> torch.Tensor:
    """Desired unit sweep direction in the robot base XY plane."""
    command = env.command_manager.get_command(command_name)
    return torch.cat(
        (
            command[:, :2],
            torch.zeros(env.num_envs, 1, device=env.device),
        ),
        dim=-1,
    )
