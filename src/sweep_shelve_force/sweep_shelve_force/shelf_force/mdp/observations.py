"""41-D Cartesian policy observation for shelf force control."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation


def end_effector_pose_b(env, asset_cfg) -> torch.Tensor:
    """EEF ``xyz + RPY`` in robot-base coordinates."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion_b)
    return torch.cat(
        (position_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
    )


def end_effector_twist_b(env, asset_cfg) -> torch.Tensor:
    """EEF linear and angular velocity in robot-base coordinates."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    relative_velocity_w = robot.data.body_vel_w[:, body_id] - robot.data.root_vel_w
    linear_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, :3]
    )
    angular_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, 3:]
    )
    return torch.cat((linear_b, angular_b), dim=-1)


def initial_target_pose_b(env, command_name: str) -> torch.Tensor:
    """Selected object's reset-time ``xyz + RPY`` pose in robot base."""
    return env.command_manager.get_term(command_name).initial_pose_b
