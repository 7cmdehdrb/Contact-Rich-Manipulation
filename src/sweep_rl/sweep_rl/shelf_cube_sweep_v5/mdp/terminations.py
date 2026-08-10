"""Failure termination terms for the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def cube_tilt_angle(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Return the angle between the Cube local-up and world-up axes.

    Measuring the up-axis directly avoids Euler roll/pitch axis-order ambiguity and
    ignores pure yaw rotation. The v5 Cube is spawned with its height along local Z.
    Invalid quaternions return infinity so they are treated as failures.
    """

    quat_norm = torch.linalg.vector_norm(root_quat_w, dim=-1, keepdim=True)
    valid = torch.isfinite(root_quat_w).all(dim=-1) & (quat_norm[:, 0] > 1.0e-8)
    safe_quat = torch.where(
        valid[:, None],
        root_quat_w / quat_norm.clamp_min(1.0e-8),
        torch.zeros_like(root_quat_w),
    )
    local_up = torch.zeros(
        (root_quat_w.shape[0], 3),
        dtype=root_quat_w.dtype,
        device=root_quat_w.device,
    )
    local_up[:, 2] = 1.0
    up_z = math_utils.quat_apply(safe_quat, local_up)[:, 2]
    angle = torch.acos(torch.clamp(up_z, -1.0, 1.0))
    return torch.where(valid, angle, torch.full_like(angle, torch.inf))


def cube_below_minimum_height(
    env,
    minimum_center_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the Cube center falls below the active shelf surface."""

    if not math.isfinite(minimum_center_height):
        raise ValueError("Minimum Cube center height must be finite.")
    target: RigidObject = env.scene[object_cfg.name]
    height = target.data.root_pos_w[:, 2]
    return ~torch.isfinite(height) | (height < minimum_center_height)


def cube_excessive_tilt(
    env,
    maximum_tilt: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the Cube up-axis tilt exceeds the configured angle."""

    if not 0.0 < maximum_tilt < math.pi:
        raise ValueError("Maximum Cube tilt must be between zero and pi.")
    target: RigidObject = env.scene[object_cfg.name]
    return cube_tilt_angle(target.data.root_quat_w) > maximum_tilt


def cube_too_fast(
    env,
    maximum_linear_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the Cube linear speed exceeds the configured limit."""

    if maximum_linear_speed <= 0.0:
        raise ValueError("Maximum Cube linear speed must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    velocity = target.data.root_lin_vel_w
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    return ~torch.isfinite(velocity).all(dim=-1) | (speed > maximum_linear_speed)


def arm_joint_too_fast(
    env,
    maximum_joint_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any selected arm joint exceeds the speed limit."""

    if maximum_joint_speed <= 0.0:
        raise ValueError("Maximum arm joint speed must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    joint_velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return ~torch.isfinite(joint_velocity).all(dim=-1) | torch.any(
        torch.abs(joint_velocity) > maximum_joint_speed, dim=-1
    )
