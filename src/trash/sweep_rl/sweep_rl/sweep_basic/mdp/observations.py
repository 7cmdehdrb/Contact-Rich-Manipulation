"""Noise-free robot-base-frame observations for the sweep policy."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def _pose_in_robot_base(
    robot: Articulation, position_w: torch.Tensor, quaternion_w: torch.Tensor
) -> torch.Tensor:
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position_w,
        quaternion_w,
    )
    return torch.cat((position_b, math_utils.quat_unique(quaternion_b)), dim=-1)


def end_effector_pose_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    frames: FrameTransformer = env.scene[frame_cfg.name]
    return _pose_in_robot_base(
        robot,
        frames.data.target_pos_w[:, frame_index],
        frames.data.target_quat_w[:, frame_index],
    )


def target_position_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
    )
    return position_b


def target_linear_velocity_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Target world linear velocity rotated into the robot-base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    return math_utils.quat_apply_inverse(
        robot.data.root_quat_w, target.data.root_lin_vel_w
    )


def goal_position_b(env, command_name: str) -> torch.Tensor:
    """Return the fixed target goal position in robot-base coordinates."""
    command = env.command_manager.get_term(command_name)
    return command.goal_pos_b
