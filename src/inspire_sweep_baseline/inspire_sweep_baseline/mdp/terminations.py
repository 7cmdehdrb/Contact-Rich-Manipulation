"""Local shelf-sweep termination rules; Arm cutoff is independent of fingers."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def object_drop(env, height_condition=1.04, rotation_condition=0.9):
    state = env.scene["object_collection"].data.object_link_state_w
    # Source threshold is in each environment's local shelf coordinate system.
    height = state[..., 2] - env.scene.env_origins[:, None, 2]
    count, objects = state.shape[:2]
    roll, pitch, _ = euler_xyz_from_quat(state[..., 3:7].reshape(-1, 4))
    roll = ((roll + torch.pi) % (2.0 * torch.pi) - torch.pi).reshape(count, objects)
    pitch = ((pitch + torch.pi) % (2.0 * torch.pi) - torch.pi).reshape(count, objects)
    return torch.any(
        (height < height_condition) | (roll.abs() > rotation_condition) | (pitch.abs() > rotation_condition),
        dim=1,
    )


def push_fast(env, speed_condition=0.3):
    rows = torch.arange(env.num_envs, device=env.device)
    velocity = env.scene["object_collection"].data.object_lin_vel_w[rows, env.target_id[:, 0].long()]
    return torch.linalg.vector_norm(velocity, dim=-1) > speed_condition


def shelf_collision(env, threshold=0.1):
    shelf = env.scene["shelf"].data
    height = shelf.root_pos_w[:, 2] + 1.05
    fingers = env.scene["finger_frame"].data.target_pos_w[:, :2, 2] - height[:, None]
    wrist = env.scene["wrist_frame"].data.target_pos_w[:, 0, 2] - height
    return (
        (torch.linalg.vector_norm(shelf.root_vel_w, dim=-1) > threshold)
        | torch.any(fingers < 0.01, dim=1)
        | (wrist < 0.07)
    )


def arm_velocity(env, asset_cfg=SceneEntityCfg("robot"), threshold=1.0):
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return torch.any(velocity.abs() > threshold, dim=1)
