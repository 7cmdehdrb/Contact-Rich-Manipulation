"""Implicit Reach-to-Sweep reward terms for the inherited Cube task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
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
    target_velocity: torch.Tensor,
    gate_distance: float,
    velocity_reward_min: float = 0.05,
    velocity_penalty_min: float = 0.10,
) -> torch.Tensor:
    """Compute the piecewise raw pushing reward from a velocity measurement."""
    if gate_distance <= 0.0:
        raise ValueError("Sweep gate distance must be positive.")
    if not 0.0 <= velocity_reward_min < velocity_penalty_min:
        raise ValueError(
            "Velocity reward threshold must be non-negative and smaller than "
            "the velocity penalty threshold."
        )
    zeta_m = (contact_distance < gate_distance) & (wrist_y_error < gate_distance)
    target_speed = torch.abs(target_velocity)
    object_velocity_reward = torch.where(
        target_speed > velocity_reward_min,
        torch.where(target_speed < velocity_penalty_min, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * torch.exp(-5.0 * distance),
        zeta_m.float() * ((1.0 - distance / 0.18) + object_velocity_reward),
    )


def cube_upright_quality(
    root_quat_w: torch.Tensor,
    max_tilt_radians: float,
) -> torch.Tensor:
    """Return a smooth [0, 1] quality that reaches zero at max tilt."""
    if not 0.0 < max_tilt_radians < 0.5 * torch.pi:
        raise ValueError("Maximum Cube tilt must be between 0 and pi/2 radians.")
    local_up = torch.zeros(
        (root_quat_w.shape[0], 3),
        dtype=root_quat_w.dtype,
        device=root_quat_w.device,
    )
    local_up[:, 2] = 1.0
    up_axis_w = math_utils.quat_apply(root_quat_w, local_up)
    min_up_z = torch.cos(
        torch.as_tensor(
            max_tilt_radians,
            dtype=root_quat_w.dtype,
            device=root_quat_w.device,
        )
    )
    quality = torch.clamp(
        (up_axis_w[:, 2] - min_up_z) / (1.0 - min_up_z),
        min=0.0,
        max=1.0,
    )
    return torch.where(up_axis_w[:, 2] <= min_up_z + 1.0e-6, 0.0, quality)


def pushing_target(
    env,
    command_name: str,
    cube_width: float,
    x_offset: float,
    behind_width_scale: float,
    z_offset: float,
    gate_distance: float,
    max_tilt_radians: float | None = None,
    velocity_reward_min: float = 0.05,
    velocity_penalty_min: float = 0.10,
    velocity_measurement: str = "com_y",
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
    if velocity_measurement == "com_y":
        target_velocity = target.data.root_lin_vel_w[:, 1]
    elif velocity_measurement == "root_link_planar":
        # Measure translation at the actor-frame origin.  With a bottom-heavy
        # object, COM velocity also contains omega x r and can reward rotation.
        target_velocity = torch.linalg.vector_norm(
            target.data.root_link_lin_vel_w[:, :2], dim=-1
        )
    else:
        raise ValueError(
            "Unsupported push velocity measurement "
            f"'{velocity_measurement}'. Expected 'com_y' or 'root_link_planar'."
        )

    reward = pushing_target_raw_reward(
        distance,
        contact_distance,
        wrist_y_error,
        target_velocity,
        gate_distance,
        velocity_reward_min,
        velocity_penalty_min,
    )
    if max_tilt_radians is not None:
        reward = reward * cube_upright_quality(
            target.data.root_quat_w,
            max_tilt_radians,
        )
    return reward
