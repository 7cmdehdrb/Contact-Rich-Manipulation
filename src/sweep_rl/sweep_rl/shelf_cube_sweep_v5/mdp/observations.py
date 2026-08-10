"""Observation terms owned by the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def target_object_position_b(
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


def cube_width(env, width: float) -> torch.Tensor:
    return torch.full((env.num_envs, 1), width, device=env.device)


def current_ee_pose_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return current TCP position and quaternion in the robot-base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[frame_cfg.name]
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0],
        ee_frame.data.target_quat_w[:, 0],
    )
    return torch.cat((position_b, math_utils.quat_unique(quaternion_b)), dim=-1)


def sweep_goal_position_b(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the fixed episode Sweep goal in the robot-base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_term(command_name)
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command.goal_pos_w,
    )
    return position_b
