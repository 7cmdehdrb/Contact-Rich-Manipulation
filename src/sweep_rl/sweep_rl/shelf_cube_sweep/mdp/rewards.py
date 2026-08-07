"""Implicit Reach-to-Sweep reward terms for the inherited Cube task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def cube_push_offset_position(
    target_position_w: torch.Tensor,
    cube_width: float,
    x_offset: float,
    behind_width_scale: float,
    z_offset: float,
) -> torch.Tensor:
    """Return the moving +Y push point relative to the current Cube."""
    target_offset_w = target_position_w.clone()
    target_offset_w[:, 0] += x_offset
    target_offset_w[:, 1] -= cube_width * behind_width_scale
    target_offset_w[:, 2] += z_offset
    return target_offset_w


def pushing_target_raw_reward(
    distance: torch.Tensor,
    contact_distance: torch.Tensor,
    wrist_y_error: torch.Tensor,
    target_y_velocity: torch.Tensor,
) -> torch.Tensor:
    """Compute the requested piecewise raw +Y pushing reward."""
    zeta_m = (contact_distance < 0.04) & (wrist_y_error < 0.04)
    target_y_speed = torch.abs(target_y_velocity)
    object_velocity_reward = torch.where(
        target_y_speed > 0.05,
        torch.where(target_y_speed < 0.10, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * torch.exp(-5.0 * distance),
        zeta_m.float() * ((1.0 - distance / 0.18) + object_velocity_reward),
    )


def pushing_target(
    env,
    command_name: str,
    cube_width: float,
    x_offset: float,
    behind_width_scale: float,
    z_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
) -> torch.Tensor:
    """Gate Sweep shaping with the current Cube-relative contact point."""
    command_term = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    wrist_frame: FrameTransformer = env.scene[wrist_frame_cfg.name]

    target_offset_w = cube_push_offset_position(
        target.data.root_pos_w,
        cube_width,
        x_offset,
        behind_width_scale,
        z_offset,
    )
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]
    contact_distance = torch.linalg.norm(target_offset_w - ee_pos_w, dim=-1)
    distance = torch.linalg.norm(
        command_term.goal_pos_w - target.data.root_pos_w, dim=-1
    )
    wrist_y_error = torch.abs(
        target_offset_w[:, 1] - wrist_frame.data.target_pos_w[:, 0, 1]
    )
    return pushing_target_raw_reward(
        distance,
        contact_distance,
        wrist_y_error,
        target.data.root_lin_vel_w[:, 1],
    )
