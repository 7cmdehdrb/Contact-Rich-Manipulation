"""Sweep-Policy observations adapted to one direct RigidObject."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def joint_pos_rel_arm(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return default-relative positions for the six UR5e arm joints only."""

    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, :6] - robot.data.default_joint_pos[:, :6]


def joint_vel_rel_arm(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, :6] - robot.data.default_joint_vel[:, :6]


def target_position_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target.data.root_pos_w
    )
    return position_b


def target_width(env, width: float) -> torch.Tensor:
    return torch.full((env.num_envs, 1), width, device=env.device)


def end_effector_pose_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[frame_cfg.name]
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0],
        ee_frame.data.target_quat_w[:, 0],
    )
    return torch.cat((position_b, math_utils.quat_unique(quaternion_b)), dim=-1)


def goal_position_b(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.goal_pos_b
