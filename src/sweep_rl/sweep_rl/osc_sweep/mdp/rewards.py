"""Direction, distance, and force-centered reward formulation."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .common import (
    active_gripper_side_direction_b,
    desired_direction_b,
    object_displacement_b,
    side_pad_contact_quality,
    target_contact_data_w,
)


def reaching_precontact_pose(
    env,
    command_name: str,
    std: float,
    stand_off: float,
    eef_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward approaching behind the cube along the requested push direction."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    initial_pos_b = command.initial_pose_b[:, :3]
    direction_b = desired_direction_b(env, command_name)
    precontact_b = initial_pos_b - stand_off * direction_b
    precontact_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        precontact_b,
    )
    distance = torch.linalg.norm(eef_pos_w - precontact_w, dim=-1)
    return torch.exp(-torch.square(distance / std))


def object_velocity_along_direction(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Signed object velocity along the desired world-frame direction."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    return torch.sum(
        target.data.root_lin_vel_w * command.direction_w, dim=-1
    )


def normalized_progress(
    env,
    command_name: str,
) -> torch.Tensor:
    """Current longitudinal displacement divided by desired sweep length."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    length = env.command_manager.get_command(command_name)[:, 2]
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    return torch.clamp(progress / torch.clamp(length, min=1.0e-6), -1.0, 1.2)


def direction_alignment(
    env,
    command_name: str,
    minimum_displacement: float = 0.005,
) -> torch.Tensor:
    """Cosine alignment between actual XY displacement and desired direction."""
    displacement_b = object_displacement_b(env, command_name)
    displacement_xy = displacement_b[:, :2]
    norm = torch.linalg.norm(displacement_xy, dim=-1)
    direction_xy = env.command_manager.get_command(command_name)[:, :2]
    cosine = torch.sum(displacement_xy * direction_xy, dim=-1) / torch.clamp(
        norm, min=1.0e-6
    )
    return torch.where(
        norm > minimum_displacement, torch.clamp(cosine, -1.0, 1.0), 0.0
    )


def lateral_displacement(
    env,
    command_name: str,
) -> torch.Tensor:
    """Normalized displacement perpendicular to the desired direction."""
    displacement_b = object_displacement_b(env, command_name)
    direction = env.command_manager.get_command(command_name)[:, :2]
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    lateral = torch.abs(
        torch.sum(displacement_b[:, :2] * perpendicular, dim=-1)
    )
    length = env.command_manager.get_command(command_name)[:, 2]
    return lateral / torch.clamp(length, min=1.0e-6)


def endpoint_tracking(
    env,
    command_name: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Gaussian reward for stopping at the requested sweep length."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    return torch.exp(-torch.square(error / std))


def overshoot_penalty(env, command_name: str) -> torch.Tensor:
    """Penalty for longitudinal motion beyond the desired length."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    command = env.command_manager.get_command(command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    return torch.relu(progress - command[:, 2]) / torch.clamp(
        command[:, 2], min=1.0e-6
    )


def target_force_tracking(
    env,
    command_name: str,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
) -> torch.Tensor:
    """Track requested target-contact force magnitude within its sampled band."""
    _, force_w, contact_mask = target_contact_data_w(env, sensor_names)
    measured_force = torch.linalg.norm(force_w, dim=-1)
    command = env.command_manager.get_command(command_name)
    desired_force = command[:, 3]
    tolerance = torch.clamp(command[:, 4], min=1.0e-3)
    tracking = torch.exp(
        -torch.square((measured_force - desired_force) / tolerance)
    )
    # Force is meaningful only while the gripper is touching the target.
    return tracking * contact_mask.float()


def gripper_side_direction_alignment(
    env,
    command_name: str,
    side_axis_local: tuple[float, float, float],
    proximity_std: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Align the active broad-side normal with the requested sweep direction."""
    if proximity_std <= 0.0:
        raise ValueError("proximity_std must be positive.")

    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    active_side_b = active_gripper_side_direction_b(
        env,
        eef_cfg=eef_cfg,
        object_cfg=object_cfg,
        side_axis_local=side_axis_local,
    )
    desired_b = desired_direction_b(env, command_name)
    side_xy = active_side_b[:, :2]
    side_xy = side_xy / torch.clamp(
        torch.linalg.norm(side_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    alignment = torch.sum(side_xy * desired_b[:, :2], dim=-1)

    eef_pos_w = robot.data.body_pos_w[:, eef_body_id]
    distance = torch.linalg.norm(target.data.root_pos_w - eef_pos_w, dim=-1)
    proximity = torch.exp(-torch.square(distance / proximity_std))
    return torch.clamp(alignment, -1.0, 1.0) * proximity


def side_pad_center_contact(
    env,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Reward target contact at the center of either broad gripper side pad."""
    quality, _ = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return quality


def off_center_target_contact(
    env,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Penalize target contact on pad edges, narrow faces, or inner faces."""
    quality, contact_mask = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return contact_mask.float() * (1.0 - quality)


def side_target_force_tracking(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
) -> torch.Tensor:
    """Track desired force only through broad-side central pad contact."""
    _, force_w, _ = target_contact_data_w(env, sensor_names)
    measured_force = torch.linalg.norm(force_w, dim=-1)
    command = env.command_manager.get_command(command_name)
    desired_force = command[:, 3]
    tolerance = torch.clamp(command[:, 4], min=1.0e-3)
    tracking = torch.exp(
        -torch.square((measured_force - desired_force) / tolerance)
    )
    quality, _ = side_pad_contact_quality(
        env,
        sensor_names=sensor_names,
        pad_size=pad_size,
        face_normal_axis=face_normal_axis,
        center_sigma=center_sigma,
        face_sigma=face_sigma,
    )
    return tracking * quality


def target_contact_bonus(
    env,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
) -> torch.Tensor:
    """Binary reward for making target-specific contact."""
    _, _, contact_mask = target_contact_data_w(env, sensor_names)
    return contact_mask.float()


def ft_torque_excess(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize F/T torque above a safe free-space/load threshold."""
    robot: Articulation = env.scene[asset_cfg.name]
    wrench = -robot.data.body_incoming_joint_wrench_b[
        :, asset_cfg.body_ids[0], :
    ]
    torque_norm = torch.linalg.norm(wrench[:, 3:6], dim=-1)
    return torch.relu(torque_norm - threshold)


def commanded_joint_effort_l2(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Normalized OSC torque penalty."""
    action = env.action_manager.get_term(action_name)
    effort_limits = action._asset.data.joint_effort_limits[:, action._joint_ids]
    normalized = action.joint_efforts / torch.clamp(
        effort_limits, min=1.0e-6
    )
    return torch.sum(torch.square(normalized), dim=-1)


def torque_saturation(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalty indicator for invalid/clipped actions or saturated OSC torque."""
    return env.action_manager.get_term(action_name).torque_saturated.float()


def sweep_success_bonus(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse success bonus for correct direction and requested length."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    lateral_error = lateral_displacement(env, command_name)
    return (
        (endpoint_error < endpoint_threshold)
        & (lateral_error < lateral_threshold)
    ).float()
