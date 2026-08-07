"""Shelf-safety reward terms for Cube pre-reaching."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from sweep_rl.shelf_reach.mdp.rewards import tcp_position_command_error


def tcp_position_command_reward_exp(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """Map TCP-to-moving-offset distance to ``exp(-10 * distance)``."""
    distance = tcp_position_command_error(
        env,
        command_name,
        robot_cfg=robot_cfg,
        frame_cfg=frame_cfg,
        frame_index=frame_index,
    )
    return torch.exp(-10.0 * distance)


def shelf_floor_contact_mask(
    force_matrix_w: torch.Tensor,
    contact_pos_w: torch.Tensor,
    shelf_pos_w: torch.Tensor,
    shelf_quat_w: torch.Tensor,
    *,
    force_threshold: float,
    surface_height: float,
    surface_tolerance: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> torch.Tensor:
    """Detect filtered robot contacts on the Cube-supporting shelf board."""
    if force_threshold <= 0.0 or surface_tolerance <= 0.0:
        raise ValueError("Contact force threshold and surface tolerance must be positive.")
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("Shelf floor bounds must be strictly increasing.")
    if force_matrix_w.shape != contact_pos_w.shape or force_matrix_w.ndim != 4:
        raise ValueError(
            "Contact forces and positions must have matching (N, B, M, 3) shapes."
        )

    valid = torch.isfinite(force_matrix_w).all(dim=-1) & torch.isfinite(
        contact_pos_w
    ).all(dim=-1)
    safe_force_w = torch.nan_to_num(force_matrix_w, nan=0.0)
    safe_contact_pos_w = torch.nan_to_num(contact_pos_w, nan=0.0)

    num_bodies, num_filters = force_matrix_w.shape[1:3]
    shelf_pos_w = shelf_pos_w[:, None, None, :]
    shelf_quat_w = shelf_quat_w[:, None, None, :].expand(
        -1, num_bodies, num_filters, -1
    )
    contact_pos_s = math_utils.quat_apply_inverse(
        shelf_quat_w.reshape(-1, 4),
        (safe_contact_pos_w - shelf_pos_w).reshape(-1, 3),
    ).reshape_as(safe_contact_pos_w)

    on_floor = (
        (contact_pos_s[..., 0] >= x_bounds[0])
        & (contact_pos_s[..., 0] <= x_bounds[1])
        & (contact_pos_s[..., 1] >= y_bounds[0])
        & (contact_pos_s[..., 1] <= y_bounds[1])
        & (torch.abs(contact_pos_s[..., 2] - surface_height) <= surface_tolerance)
    )
    force = torch.linalg.norm(safe_force_w, dim=-1)
    return torch.any(valid & on_floor & (force > force_threshold), dim=(1, 2))


def shelf_collision(
    env,
    sensor_name: str,
    force_threshold: float,
    surface_height: float,
    surface_tolerance: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
) -> torch.Tensor:
    """Flag UR5e/Gripper contact with only the Cube-supporting shelf floor."""

    shelf: RigidObject = env.scene[shelf_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    contact_pos_w = sensor.data.contact_pos_w
    if force_matrix_w is None or contact_pos_w is None:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' must track filtered forces and contact points."
        )
    return shelf_floor_contact_mask(
        force_matrix_w,
        contact_pos_w,
        shelf.data.root_pos_w,
        shelf.data.root_quat_w,
        force_threshold=force_threshold,
        surface_height=surface_height,
        surface_tolerance=surface_tolerance,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    ).to(dtype=torch.float32)
