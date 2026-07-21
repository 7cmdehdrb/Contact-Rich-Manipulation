"""Exactly four requested reward categories for sweep and Home return."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .common import (
    PHASE_HOME,
    PHASE_REACH,
    PHASE_SWEEP,
    filtered_contact_mask,
    object_displacement_b,
    target_contact_data_w,
)
from .events import TARGET_SIZE_BUFFER


def reaching(
    env,
    command_name: str,
    distance_std: float,
    surface_clearance: float,
    table_side_pad_offset: float,
    sensor_names: tuple[str, ...],
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Guide the EEF to the size-aware pre-contact pose before first contact."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, TARGET_SIZE_BUFFER):
        raise RuntimeError("Target-size buffer is unavailable.")
    stand_off = 0.5 * getattr(env, TARGET_SIZE_BUFFER) + surface_clearance
    precontact_w = target.data.root_pos_w - stand_off.unsqueeze(-1) * command.direction_w
    precontact_w[:, 2] += table_side_pad_offset
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    distance = torch.linalg.norm(eef_pos_w - precontact_w, dim=-1)
    _, _, contact = target_contact_data_w(env, sensor_names=sensor_names)
    normalized = torch.clamp(distance / distance_std, max=3.0)
    value = torch.exp(-torch.square(normalized)) - 0.20 * normalized
    return value * (command.task_phase == PHASE_REACH).float() * (~contact).float()


def contact(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Reward target-specific contact throughout the sweep phase."""
    command = env.command_manager.get_term(command_name)
    _, _, contact_mask = target_contact_data_w(
        env, sensor_names=sensor_names, force_threshold=force_threshold
    )
    return contact_mask.float() * (command.task_phase == PHASE_SWEEP).float()


def push(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    velocity_std: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    endpoint_std: float,
    stopped_speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Privileged direction, distance, and velocity-profile tracking reward."""
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    displacement_b = object_displacement_b(env, command_name)
    progress = torch.sum(displacement_b[:, :2] * command[:, :2], dim=-1)
    lateral_vector = displacement_b[:, :2] - progress.unsqueeze(-1) * command[:, :2]
    lateral = torch.linalg.norm(lateral_vector, dim=-1)
    remaining = command[:, 2] - progress

    acceleration_factor = torch.clamp(progress / acceleration_distance, 0.0, 1.0)
    acceleration_factor = acceleration_factor * acceleration_factor * (
        3.0 - 2.0 * acceleration_factor
    )
    acceleration_factor = initial_speed_fraction + (
        1.0 - initial_speed_fraction
    ) * acceleration_factor
    stopping_factor = torch.clamp(remaining / stopping_distance, 0.0, 1.0)
    stopping_factor = stopping_factor * stopping_factor * (3.0 - 2.0 * stopping_factor)
    desired_speed = command[:, 3] * acceleration_factor * stopping_factor
    desired_velocity_w = term.direction_w * desired_speed.unsqueeze(-1)
    velocity_error = torch.linalg.norm(
        target.data.root_lin_vel_w - desired_velocity_w, dim=-1
    )
    velocity_tracking = torch.exp(-torch.square(velocity_error / velocity_std))
    _, _, contact_mask = target_contact_data_w(env, sensor_names=sensor_names)

    forward_speed = torch.sum(target.data.root_lin_vel_w * term.direction_w, dim=-1)
    direction_progress = torch.clamp(
        forward_speed / torch.clamp(command[:, 3], min=1.0e-6), -1.0, 1.0
    )
    normalized_lateral = lateral / torch.clamp(command[:, 2], min=1.0e-6)
    overshoot = torch.relu(progress - command[:, 2]) / torch.clamp(
        command[:, 2], min=1.0e-6
    )
    endpoint_error = torch.linalg.norm(target.data.root_pos_w - term.goal_pos_w, dim=-1)
    object_speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    stopped_at_goal = torch.exp(
        -torch.square(endpoint_error / endpoint_std)
        - torch.square(object_speed / stopped_speed_std)
    )

    value = (
        2.5 * velocity_tracking * contact_mask.float()
        + 0.75 * direction_progress
        + 4.0 * stopped_at_goal
        - 2.0 * normalized_lateral
        - 3.0 * overshoot
    )
    return value * (term.task_phase == PHASE_SWEEP).float()


def home_return(
    env,
    command_name: str,
    joint_std: float,
    joint_error_scale: float,
    contact_sensor_name: str | tuple[str, ...],
    contact_force_threshold: float,
    displacement_scale: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward OSC homing while keeping the parked target untouched."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    joint_error = math_utils.wrap_to_pi(
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    mean_square_error = torch.mean(torch.square(joint_error), dim=-1)
    pose_tracking = torch.exp(-mean_square_error / (joint_std * joint_std))
    normalized_joint_error = torch.mean(torch.abs(joint_error), dim=-1) / joint_error_scale
    contact_mask = filtered_contact_mask(
        env, contact_sensor_name, contact_force_threshold
    )
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    disturbance = torch.clamp(displacement / displacement_scale, max=4.0)
    value = (
        3.0 * pose_tracking
        - normalized_joint_error
        - 4.0 * contact_mask.float()
        - 3.0 * disturbance
    )
    return value * (command.task_phase == PHASE_HOME).float()
