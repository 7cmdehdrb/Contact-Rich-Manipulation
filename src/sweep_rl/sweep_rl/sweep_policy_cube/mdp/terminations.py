"""Fixed Sweep-Policy terminations adapted to a single Cube."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


def drop_object_termination(
    env,
    height_condition: float,
    rotation_condition: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    target: RigidObject = env.scene[object_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(target.data.root_quat_w)
    is_dropped = target.data.root_pos_w[:, 2] < height_condition
    is_flipped = (torch.abs(wrap_to_pi(roll)) > rotation_condition) | (
        torch.abs(wrap_to_pi(pitch)) > rotation_condition
    )
    return is_dropped | is_flipped


def push_fast_termination(
    env,
    speed_condition: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    target: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.norm(target.data.root_lin_vel_w, dim=-1) > speed_condition


def shelf_collision_termination(
    env,
    threshold: float,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]
    fingers: FrameTransformer = env.scene[finger_frame_cfg.name]
    wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
    shelf_surface_z = shelf.data.root_pos_w[:, 2] + 1.05
    left_clearance = fingers.data.target_pos_w[:, 0, 2] - shelf_surface_z
    right_clearance = fingers.data.target_pos_w[:, 1, 2] - shelf_surface_z
    wrist_clearance = wrist.data.target_pos_w[:, 0, 2] - shelf_surface_z
    shelf_moving = torch.linalg.norm(shelf.data.root_vel_w, dim=-1) > threshold
    return (
        shelf_moving
        | (left_clearance < 0.01)
        | (right_clearance < 0.01)
        | (wrist_clearance < 0.07)
    )


def hand_velocity_termination(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(torch.abs(robot.data.joint_vel[:, :6]) > threshold, dim=-1)
