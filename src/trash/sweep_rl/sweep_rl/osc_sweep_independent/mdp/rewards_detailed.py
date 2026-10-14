"""Phase-specific dense rewards for the inherited detailed sweep task."""

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


def _phase_mask(env, command_name: str, phase: int) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return (command.task_phase == phase).float()


def _precontact_distance(
    env,
    command_name: str,
    surface_clearance: float,
    table_side_pad_offset: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if not hasattr(env, TARGET_SIZE_BUFFER):
        raise RuntimeError("Target-size buffer is unavailable.")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    stand_off = 0.5 * getattr(env, TARGET_SIZE_BUFFER) + surface_clearance
    desired_w = target.data.root_pos_w - stand_off.unsqueeze(-1) * command.direction_w
    desired_w[:, 2] += table_side_pad_offset
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    return torch.linalg.norm(eef_pos_w - desired_w, dim=-1)


def reach_pose_tracking(
    env,
    command_name: str,
    distance_std: float,
    surface_clearance: float,
    table_side_pad_offset: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Gaussian tracking of the size-aware pose beside the object."""
    if distance_std <= 0.0:
        raise ValueError("distance_std must be positive.")
    distance = _precontact_distance(
        env,
        command_name,
        surface_clearance,
        table_side_pad_offset,
        eef_cfg,
        object_cfg,
    )
    return torch.exp(-torch.square(distance / distance_std)) * _phase_mask(
        env, command_name, PHASE_REACH
    )


def reach_pose_error(
    env,
    command_name: str,
    distance_scale: float,
    maximum_normalized_error: float,
    surface_clearance: float,
    table_side_pad_offset: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Normalized dense approach error, capped against extreme reset states."""
    if distance_scale <= 0.0 or maximum_normalized_error <= 0.0:
        raise ValueError("Reach error scales must be positive.")
    distance = _precontact_distance(
        env,
        command_name,
        surface_clearance,
        table_side_pad_offset,
        eef_cfg,
        object_cfg,
    )
    value = torch.clamp(distance / distance_scale, max=maximum_normalized_error)
    return value * _phase_mask(env, command_name, PHASE_REACH)


def sweep_contact(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Binary target-pad contact reward during Sweep only."""
    _, _, contact = target_contact_data_w(
        env, sensor_names=sensor_names, force_threshold=force_threshold
    )
    return contact.float() * _phase_mask(env, command_name, PHASE_SWEEP)


def _sweep_progress(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    displacement = object_displacement_b(env, command_name)
    progress = torch.sum(displacement[:, :2] * command[:, :2], dim=-1)
    lateral_vector = displacement[:, :2] - progress.unsqueeze(-1) * command[:, :2]
    return progress, torch.linalg.norm(lateral_vector, dim=-1)


def _desired_sweep_velocity_w(
    env,
    command_name: str,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
) -> torch.Tensor:
    if acceleration_distance <= 0.0 or stopping_distance <= 0.0:
        raise ValueError("Sweep profile distances must be positive.")
    if not 0.0 <= initial_speed_fraction <= 1.0:
        raise ValueError("initial_speed_fraction must be in [0, 1].")
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    progress, _ = _sweep_progress(env, command_name)
    remaining = command[:, 2] - progress
    acceleration = torch.clamp(progress / acceleration_distance, 0.0, 1.0)
    acceleration = acceleration * acceleration * (3.0 - 2.0 * acceleration)
    acceleration = initial_speed_fraction + (1.0 - initial_speed_fraction) * acceleration
    stopping = torch.clamp(remaining / stopping_distance, 0.0, 1.0)
    stopping = stopping * stopping * (3.0 - 2.0 * stopping)
    desired_speed = command[:, 3] * acceleration * stopping
    return term.direction_w * desired_speed.unsqueeze(-1)


def sweep_velocity_tracking(
    env,
    command_name: str,
    velocity_std: float,
    acceleration_distance: float,
    stopping_distance: float,
    initial_speed_fraction: float,
    sensor_names: tuple[str, ...],
    force_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track the accelerate-cruise-stop velocity profile while in contact."""
    if velocity_std <= 0.0:
        raise ValueError("velocity_std must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity = _desired_sweep_velocity_w(
        env,
        command_name,
        acceleration_distance,
        stopping_distance,
        initial_speed_fraction,
    )
    error = torch.linalg.norm(target.data.root_lin_vel_w - desired_velocity, dim=-1)
    _, _, contact = target_contact_data_w(
        env, sensor_names=sensor_names, force_threshold=force_threshold
    )
    return (
        torch.exp(-torch.square(error / velocity_std))
        * contact.float()
        * _phase_mask(env, command_name, PHASE_SWEEP)
    )


def sweep_forward_progress(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Signed forward object speed normalized by the commanded cruise speed."""
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    forward_speed = torch.sum(target.data.root_lin_vel_w * term.direction_w, dim=-1)
    value = torch.clamp(
        forward_speed / torch.clamp(command[:, 3], min=1.0e-6), -1.0, 1.0
    )
    return value * _phase_mask(env, command_name, PHASE_SWEEP)


def sweep_endpoint_error(
    env,
    command_name: str,
    maximum_normalized_error: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Endpoint distance normalized by the feasible sampled sweep length."""
    if maximum_normalized_error <= 0.0:
        raise ValueError("maximum_normalized_error must be positive.")
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(target.data.root_pos_w - term.goal_pos_w, dim=-1)
    value = torch.clamp(
        error / torch.clamp(command[:, 2], min=1.0e-6),
        max=maximum_normalized_error,
    )
    return value * _phase_mask(env, command_name, PHASE_SWEEP)


def sweep_lateral_error(env, command_name: str) -> torch.Tensor:
    """Normalized distance from the commanded line on the narrow shelf."""
    command = env.command_manager.get_command(command_name)
    _, lateral = _sweep_progress(env, command_name)
    value = lateral / torch.clamp(command[:, 2], min=1.0e-6)
    return value * _phase_mask(env, command_name, PHASE_SWEEP)


def sweep_overshoot(env, command_name: str) -> torch.Tensor:
    """Normalized motion beyond the sampled endpoint."""
    command = env.command_manager.get_command(command_name)
    progress, _ = _sweep_progress(env, command_name)
    value = torch.relu(progress - command[:, 2]) / torch.clamp(
        command[:, 2], min=1.0e-6
    )
    return value * _phase_mask(env, command_name, PHASE_SWEEP)


def sweep_stopped_at_goal(
    env,
    command_name: str,
    position_std: float,
    speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Joint Gaussian for precise placement and low object speed."""
    if position_std <= 0.0 or speed_std <= 0.0:
        raise ValueError("Goal tracking scales must be positive.")
    term = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    position_error = torch.linalg.norm(target.data.root_pos_w - term.goal_pos_w, dim=-1)
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    value = torch.exp(
        -torch.square(position_error / position_std) - torch.square(speed / speed_std)
    )
    return value * _phase_mask(env, command_name, PHASE_SWEEP)


def home_joint_pose_tracking(
    env,
    command_name: str,
    joint_std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gaussian canonical Home-joint tracking."""
    if joint_std <= 0.0:
        raise ValueError("joint_std must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    error = math_utils.wrap_to_pi(
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    value = torch.exp(-torch.mean(torch.square(error / joint_std), dim=-1))
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_joint_error(
    env,
    command_name: str,
    error_scale: float,
    maximum_normalized_error: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Capped mean absolute Home-joint error."""
    if error_scale <= 0.0 or maximum_normalized_error <= 0.0:
        raise ValueError("Home error scales must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    error = math_utils.wrap_to_pi(
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    value = torch.clamp(
        torch.mean(torch.abs(error), dim=-1) / error_scale,
        max=maximum_normalized_error,
    )
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_eef_clearance(
    env,
    command_name: str,
    safe_distance: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Smooth reward for withdrawing from the parked object."""
    if safe_distance <= 0.0:
        raise ValueError("safe_distance must be positive.")
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_pos_w = robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    normalized = torch.clamp(
        torch.linalg.norm(eef_pos_w - target.data.root_pos_w, dim=-1) / safe_distance,
        0.0,
        1.0,
    )
    value = normalized * normalized * (3.0 - 2.0 * normalized)
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_contact_penalty(
    env,
    command_name: str,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Penalize any robot-target contact after entering Home."""
    contact = filtered_contact_mask(env, sensor_names, force_threshold)
    return contact.float() * _phase_mask(env, command_name, PHASE_HOME)


def home_goal_hold_error(
    env,
    command_name: str,
    maximum_normalized_error: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize target drift from the commanded endpoint during Home."""
    if maximum_normalized_error <= 0.0:
        raise ValueError("maximum_normalized_error must be positive.")
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(target.data.root_pos_w - term.goal_pos_w, dim=-1)
    value = torch.clamp(
        error / torch.clamp(command[:, 2], min=1.0e-6),
        max=maximum_normalized_error,
    )
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_object_speed(
    env,
    command_name: str,
    speed_scale: float,
    maximum_normalized_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Capped parked-object speed penalty."""
    if speed_scale <= 0.0 or maximum_normalized_speed <= 0.0:
        raise ValueError("Object speed scales must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    value = torch.clamp(
        torch.linalg.norm(target.data.root_lin_vel_w, dim=-1) / speed_scale,
        max=maximum_normalized_speed,
    )
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_object_displacement(
    env,
    command_name: str,
    displacement_scale: float,
    maximum_normalized_displacement: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Capped displacement from the pose latched on Home entry."""
    if displacement_scale <= 0.0 or maximum_normalized_displacement <= 0.0:
        raise ValueError("Object displacement scales must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    value = torch.clamp(
        displacement / displacement_scale,
        max=maximum_normalized_displacement,
    )
    return value * _phase_mask(env, command_name, PHASE_HOME)


def home_time(env, command_name: str) -> torch.Tensor:
    """Unit Home running cost."""
    return _phase_mask(env, command_name, PHASE_HOME)


def home_success_bonus(
    env,
    command_name: str,
    joint_position_threshold: float,
    joint_speed_threshold: float,
    endpoint_threshold: float,
    object_speed_threshold: float,
    object_displacement_threshold: float,
    sensor_names: tuple[str, ...],
    contact_force_threshold: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Pre-dwell sparse signal matching the inherited success termination."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    joint_error = torch.abs(
        math_utils.wrap_to_pi(
            robot.data.joint_pos[:, asset_cfg.joint_ids]
            - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
    )
    joint_speed = torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids])
    endpoint_error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    object_speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    contact = filtered_contact_mask(env, sensor_names, contact_force_threshold)
    return (
        (command.task_phase == PHASE_HOME)
        & torch.all(joint_error < joint_position_threshold, dim=-1)
        & torch.all(joint_speed < joint_speed_threshold, dim=-1)
        & (endpoint_error < endpoint_threshold)
        & (object_speed < object_speed_threshold)
        & (displacement < object_displacement_threshold)
        & (~contact)
    ).float()


def ft_torque_excess(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Soft penalty below the inherited hard F/T termination."""
    robot: Articulation = env.scene[asset_cfg.name]
    wrench = -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]
    return torch.relu(torch.linalg.norm(wrench[:, 3:6], dim=-1) - threshold)


def commanded_joint_effort_l2(env, action_name: str = "arm_action") -> torch.Tensor:
    """Squared OSC torque normalized by the asset effort limits."""
    action = env.action_manager.get_term(action_name)
    limits = action._asset.data.joint_effort_limits[:, action._joint_ids]
    normalized = action.joint_efforts / torch.clamp(limits, min=1.0e-6)
    return torch.sum(torch.square(normalized), dim=-1)


def torque_saturation(env, action_name: str = "arm_action") -> torch.Tensor:
    """Indicator for clipped/invalid action or saturated OSC torque."""
    return env.action_manager.get_term(action_name).torque_saturated.float()


def failure_termination_penalty(
    env,
    term_names: tuple[str, ...],
    minimum_remaining_time: float,
) -> torch.Tensor:
    """Charge remaining horizon on unsafe failure to discourage early escape."""
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in term_names:
        failure |= env.termination_manager.get_term(term_name)
    remaining = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    remaining = torch.clamp(remaining, min=minimum_remaining_time)
    return failure.float() * remaining
