"""Policy observations specific to the shelf Cube Sweep task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def end_effector_pose_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the current TCP position and unique quaternion in robot-base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[frame_cfg.name]
    position_b, quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0],
        ee_frame.data.target_quat_w[:, 0],
    )
    return torch.cat((position_b, math_utils.quat_unique(quaternion_b)), dim=-1)
