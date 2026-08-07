"""Cube observations for the pre-reach policy."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def target_object_position_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Return the un-offset Cube center position in the robot-base frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
    )
    return position_b


def cube_width(env, width: float) -> torch.Tensor:
    """Return the physical Cube width as a one-dimensional observation."""

    return torch.full((env.num_envs, 1), width, device=env.device)
