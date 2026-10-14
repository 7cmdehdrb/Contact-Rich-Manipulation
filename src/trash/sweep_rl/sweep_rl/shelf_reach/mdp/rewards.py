"""Reach rewards evaluated at the Sweep-Policy Robotiq TCP."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def _desired_pose_w(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    return math_utils.combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )


def tcp_position_command_error(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """Return TCP-to-command position distance in world coordinates."""

    desired_pos_w, _ = _desired_pose_w(env, command_name, robot_cfg)
    frame: FrameTransformer = env.scene[frame_cfg.name]
    current_pos_w = frame.data.target_pos_w[:, frame_index]
    return torch.linalg.norm(current_pos_w - desired_pos_w, dim=-1)


def tcp_position_command_error_tanh(
    env,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """Map TCP position distance with the official Reach tanh kernel."""

    distance = tcp_position_command_error(
        env,
        command_name,
        robot_cfg=robot_cfg,
        frame_cfg=frame_cfg,
        frame_index=frame_index,
    )
    return 1.0 - torch.tanh(distance / std)


def tcp_orientation_command_error(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """Return shortest-path quaternion error between TCP and command."""

    _, desired_quat_w = _desired_pose_w(env, command_name, robot_cfg)
    frame: FrameTransformer = env.scene[frame_cfg.name]
    current_quat_w = frame.data.target_quat_w[:, frame_index]
    return math_utils.quat_error_magnitude(current_quat_w, desired_quat_w)
