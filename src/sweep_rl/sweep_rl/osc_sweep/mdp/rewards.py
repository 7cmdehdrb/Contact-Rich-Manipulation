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
    return torch.sum(target.data.root_lin_vel_w * command.direction_w, dim=-1)


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
    return torch.where(norm > minimum_displacement, torch.clamp(cosine, -1.0, 1.0), 0.0)


def lateral_displacement(
    env,
    command_name: str,
) -> torch.Tensor:
    """Normalized displacement perpendicular to the desired direction."""
    displacement_b = object_displacement_b(env, command_name)
    direction = env.command_manager.get_command(command_name)[:, :2]
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    lateral = torch.abs(torch.sum(displacement_b[:, :2] * perpendicular, dim=-1))
    length = env.command_manager.get_command(command_name)[:, 2]
    return lateral / torch.clamp(length, min=1.0e-6)


def endpoint_tracking(
    env,
    command_name: str,
    std: float,
    coarse_std: float | None = None,
    coarse_weight: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Gaussian reward for stopping at the requested sweep endpoint.

    An optional wider Gaussian supplies a useful gradient while the object is
    still far from the endpoint.  The narrow component retains the precision
    incentive close to the goal.
    """
    if std <= 0.0:
        raise ValueError("std must be positive.")
    if not 0.0 <= coarse_weight <= 1.0:
        raise ValueError("coarse_weight must be in [0, 1].")
    if coarse_weight > 0.0 and (coarse_std is None or coarse_std <= 0.0):
        raise ValueError("coarse_std must be positive when coarse_weight is used.")

    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    fine_tracking = torch.exp(-torch.square(error / std))
    if coarse_weight == 0.0:
        return fine_tracking

    coarse_tracking = torch.exp(-torch.square(error / coarse_std))
    return (1.0 - coarse_weight) * fine_tracking + coarse_weight * coarse_tracking


def overshoot_penalty(env, command_name: str) -> torch.Tensor:
    """Penalty for longitudinal motion beyond the desired length."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    command = env.command_manager.get_command(command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    return torch.relu(progress - command[:, 2]) / torch.clamp(command[:, 2], min=1.0e-6)


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
    tracking = torch.exp(-torch.square((measured_force - desired_force) / tolerance))
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
    tracking = torch.exp(-torch.square((measured_force - desired_force) / tolerance))
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
    wrench = -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]
    torque_norm = torch.linalg.norm(wrench[:, 3:6], dim=-1)
    return torch.relu(torque_norm - threshold)


def commanded_joint_effort_l2(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Normalized OSC torque penalty."""
    action = env.action_manager.get_term(action_name)
    effort_limits = action._asset.data.joint_effort_limits[:, action._joint_ids]
    normalized = action.joint_efforts / torch.clamp(effort_limits, min=1.0e-6)
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
        (endpoint_error < endpoint_threshold) & (lateral_error < lateral_threshold)
    ).float()


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def desired_velocity_profile_w(
    env,
    command_name: str,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired world velocity, remaining distance, and progress.

    Speed ramps up over the initial travel distance, stays constant through
    transit, and smoothly falls to zero over the final stopping distance.  A
    non-zero initial fraction prevents zero motion from matching the profile.
    """
    if acceleration_distance <= 0.0:
        raise ValueError("acceleration_distance must be positive.")
    if stopping_distance <= 0.0:
        raise ValueError("stopping_distance must be positive.")
    if not 0.0 < initial_speed_fraction <= 1.0:
        raise ValueError("initial_speed_fraction must be in (0, 1].")

    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    remaining = command[:, 2] - progress

    acceleration_ramp = _smoothstep01(progress / acceleration_distance)
    acceleration_factor = (
        initial_speed_fraction + (1.0 - initial_speed_fraction) * acceleration_ramp
    )
    stopping_factor = _smoothstep01(remaining / stopping_distance)
    desired_speed = command[:, 3] * acceleration_factor * stopping_factor
    desired_velocity_w = command_term.direction_w * desired_speed.unsqueeze(-1)
    return desired_velocity_w, remaining, progress


def object_velocity_profile_tracking(
    env,
    command_name: str,
    std: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track the full desired object velocity across accel/cruise/stop phases."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity_w, _, _ = desired_velocity_profile_w(
        env,
        command_name,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
    )
    velocity_error = torch.linalg.norm(
        target.data.root_lin_vel_w - desired_velocity_w, dim=-1
    )
    return torch.exp(-torch.square(velocity_error / std))


def object_stall_penalty(
    env,
    command_name: str,
    startup_grace_time: float,
    endpoint_threshold: float,
    minimum_speed_fraction: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize failing to move after startup while the goal remains ahead."""
    if startup_grace_time < 0.0:
        raise ValueError("startup_grace_time must be non-negative.")
    if endpoint_threshold <= 0.0:
        raise ValueError("endpoint_threshold must be positive.")
    if not 0.0 <= minimum_speed_fraction <= 1.0:
        raise ValueError("minimum_speed_fraction must be in [0, 1].")

    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity_w, remaining, _ = desired_velocity_profile_w(
        env,
        command_name,
        acceleration_distance=acceleration_distance,
        stopping_distance=stopping_distance,
        initial_speed_fraction=initial_speed_fraction,
    )
    desired_speed = torch.linalg.norm(desired_velocity_w, dim=-1)
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command_term.direction_w, dim=-1
    )
    speed_shortfall = torch.relu(
        minimum_speed_fraction * desired_speed - forward_speed
    ) / torch.clamp(command[:, 3], min=1.0e-6)
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    transit_mask = (elapsed >= startup_grace_time) & (remaining > endpoint_threshold)
    return speed_shortfall * transit_mask.float()


def object_linear_acceleration_penalty(
    env,
    acceleration_scale: float,
    maximum_normalized_penalty: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize abrupt object-speed changes using center-of-mass acceleration."""
    if acceleration_scale <= 0.0:
        raise ValueError("acceleration_scale must be positive.")
    if maximum_normalized_penalty <= 0.0:
        raise ValueError("maximum_normalized_penalty must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    acceleration = target.data.body_com_lin_acc_w[:, 0, :]
    normalized = torch.linalg.norm(acceleration, dim=-1) / acceleration_scale
    return torch.clamp(torch.square(normalized), max=maximum_normalized_penalty)


def stopped_at_goal_reward(
    env,
    command_name: str,
    position_std: float,
    speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward accurate endpoint placement only when the object is nearly still."""
    if position_std <= 0.0 or speed_std <= 0.0:
        raise ValueError("position_std and speed_std must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    position_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return torch.exp(
        -torch.square(position_error / position_std) - torch.square(speed / speed_std)
    )


def stopped_sweep_success_bonus(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse success signal requiring both endpoint accuracy and stopping."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return (
        (endpoint_error < endpoint_threshold)
        & (lateral_displacement(env, command_name) < lateral_threshold)
        & (speed < speed_threshold)
    ).float()
