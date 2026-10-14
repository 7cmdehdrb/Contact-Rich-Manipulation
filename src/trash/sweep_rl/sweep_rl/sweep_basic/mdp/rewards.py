"""Reward terms for the fixed-direction single-stage sweep task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .reward_math import (
    gaussian_distance,
    pushing_target_kernel,
    remaining_horizon_failure_kernel,
    shelf_clearance_kernel,
    signed_square_alignment_score,
)
from .task_math import midpoint_no_entry_mask, update_contact_entry_state


def reward_for_hand_reaching(
    env,
    command_name: str,
    distance_std: float,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    distance = torch.linalg.norm(
        command.contact_point_pos_w() - command.push_point_w(), dim=-1
    )
    return gaussian_distance(distance, std=distance_std)


def align_ee_target(
    env,
    command_name: str,
) -> torch.Tensor:
    """Align TCP Y with shelf Z using a signed-square orientation score."""
    command = env.command_manager.get_term(command_name)
    return signed_square_alignment_score(command.upright_alignment())


def pushing_target(
    env,
    command_name: str,
    contact_position_threshold: float,
    wrist_y_threshold: float,
    goal_threshold: float,
    distance_scale: float,
    velocity_reward_min: float,
    velocity_penalty_min: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    if contact_position_threshold <= 0.0 or wrist_y_threshold <= 0.0:
        raise ValueError("Pushing pose thresholds must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    goal_distance = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    push_point_w = command.push_point_w()
    push_pose_distance = torch.linalg.norm(
        command.contact_point_pos_w() - push_point_w, dim=-1
    )
    wrist_y_error = torch.abs(command.wrist_pos_w()[:, 1] - push_point_w[:, 1])
    motion_gate = (
        (push_pose_distance < contact_position_threshold)
        & (wrist_y_error < wrist_y_threshold)
    )
    target_y_speed = torch.abs(target.data.root_lin_vel_w[:, 1])
    return pushing_target_kernel(
        goal_distance,
        motion_gate,
        target_y_speed,
        goal_threshold=goal_threshold,
        distance_scale=distance_scale,
        velocity_reward_min=velocity_reward_min,
        velocity_penalty_min=velocity_penalty_min,
    )


def target_contact_bonus(
    env,
    command_name: str,
) -> torch.Tensor:
    """Reward contact with any real gripper body, without pad observations."""
    command = env.command_manager.get_term(command_name)
    return command.target_contact().float()


def contact_forward_progress(
    env,
    command_name: str,
    maximum_normalized_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward commanded-direction object motion only during real contact."""
    if maximum_normalized_speed <= 0.0:
        raise ValueError("maximum_normalized_speed must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command.direction_w, dim=-1
    )
    normalized_speed = torch.clamp(
        forward_speed / command.cfg.target_speed,
        0.0,
        maximum_normalized_speed,
    )
    return command.target_contact().float() * normalized_speed


def object_velocity_tracking(
    env,
    command_name: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track the fixed velocity while giving zero reward to stationary objects."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    desired_velocity = command.direction_w * command.cfg.target_speed
    velocity_error = torch.linalg.norm(
        target.data.root_lin_vel_w - desired_velocity, dim=-1
    )
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command.direction_w, dim=-1
    )
    movement_gate = torch.clamp(
        forward_speed / command.cfg.target_speed, 0.0, 1.0
    )
    tracking = gaussian_distance(velocity_error, std=std) * movement_gate
    return tracking


def normalized_endpoint_error(
    env,
    command_name: str,
    maximum_error: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Return goal distance normalized by the commanded sweep length."""
    if maximum_error <= 0.0:
        raise ValueError("maximum_error must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    normalized = torch.clamp(
        error / command.cfg.sweep_distance, max=maximum_error
    )
    return normalized


def stopped_at_goal(
    env,
    command_name: str,
    position_std: float,
    speed_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward an accurate, stationary target at the fixed endpoint."""
    if position_std <= 0.0 or speed_std <= 0.0:
        raise ValueError("position_std and speed_std must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    position_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    score = torch.exp(
        -torch.square(position_error / position_std)
        - torch.square(speed / speed_std)
    )
    return score


def sweep_success_bonus(
    env,
    command_name: str,
    endpoint_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Sparse bonus for reaching the fixed endpoint and stopping."""
    if endpoint_threshold <= 0.0 or speed_threshold <= 0.0:
        raise ValueError("Success thresholds must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    success = (endpoint_error < endpoint_threshold) & (speed < speed_threshold)
    return success.float()


def lateral_displacement(
    env,
    command_name: str,
) -> torch.Tensor:
    """Measure target displacement orthogonal to the fixed sweep direction."""
    command = env.command_manager.get_term(command_name)
    displacement = command.target.data.root_pos_w - command.initial_pos_w
    longitudinal = torch.sum(displacement * command.direction_w, dim=-1, keepdim=True)
    lateral = displacement - longitudinal * command.direction_w
    return torch.linalg.norm(lateral, dim=-1) / command.cfg.sweep_distance


def overshoot_penalty(
    env,
    command_name: str,
) -> torch.Tensor:
    """Penalize longitudinal travel beyond the requested endpoint."""
    command = env.command_manager.get_term(command_name)
    displacement = command.target.data.root_pos_w - command.initial_pos_w
    progress = torch.sum(displacement * command.direction_w, dim=-1)
    return torch.relu(progress - command.cfg.sweep_distance) / command.cfg.sweep_distance


def object_stall_penalty(
    env,
    command_name: str,
    startup_grace_time: float,
    endpoint_threshold: float,
    minimum_speed_fraction: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize stationary transit states after an initial reach grace period."""
    if startup_grace_time < 0.0 or endpoint_threshold <= 0.0:
        raise ValueError("Invalid stall timing or endpoint threshold.")
    if not 0.0 <= minimum_speed_fraction <= 1.0:
        raise ValueError("minimum_speed_fraction must be in [0, 1].")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    displacement = target.data.root_pos_w - command.initial_pos_w
    progress = torch.sum(displacement * command.direction_w, dim=-1)
    remaining = command.cfg.sweep_distance - progress
    forward_speed = torch.sum(
        target.data.root_lin_vel_w * command.direction_w, dim=-1
    )
    shortfall = torch.relu(
        minimum_speed_fraction * command.cfg.target_speed - forward_speed
    ) / command.cfg.target_speed
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    active = (elapsed >= startup_grace_time) & (remaining > endpoint_threshold)
    return shortfall * active.float()


class MidpointNoEntryPenalty(ManagerTermBase):
    """Penalize environments missing sustained real target contact by mid-episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._contact_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._entered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contact_elapsed[env_ids] = 0.0
        self._entered[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        contact_dwell_time: float,
        episode_fraction: float,
    ) -> torch.Tensor:
        if contact_dwell_time <= 0.0:
            raise ValueError("contact_dwell_time must be positive.")
        if not 0.0 < episode_fraction < 1.0:
            raise ValueError("episode_fraction must be in (0, 1).")
        command = env.command_manager.get_term(command_name)
        contact = command.target_contact()
        self._contact_elapsed, self._entered = update_contact_entry_state(
            self._contact_elapsed,
            self._entered,
            contact,
            dt=env.step_dt,
            dwell_time=contact_dwell_time,
        )
        return midpoint_no_entry_mask(
            env.episode_length_buf,
            self._entered,
            max_episode_length=env.max_episode_length,
            episode_fraction=episode_fraction,
        ).float()


def remaining_horizon_failure_penalty(
    env,
    term_names: tuple[str, ...],
    minimum_penalty_time: float,
) -> torch.Tensor:
    """Make early safety termination cost the rollout time it avoids."""
    if minimum_penalty_time < 0.0:
        raise ValueError("minimum_penalty_time must be non-negative.")
    if not term_names:
        raise ValueError("term_names must not be empty.")
    failure = torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    for term_name in term_names:
        failure |= env.termination_manager.get_term(term_name).bool()
    return remaining_horizon_failure_kernel(
        failure,
        env.episode_length_buf,
        max_episode_length=env.max_episode_length,
        minimum_steps=minimum_penalty_time / env.step_dt,
    )


def shelf_collision(
    env,
    surface_height: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    finger_clearance: float,
    wrist_clearance: float,
    left_finger_frame_index: int,
    right_finger_frame_index: int,
    wrist_frame_index: int,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]
    frames: FrameTransformer = env.scene[frame_cfg.name]
    indices = [
        left_finger_frame_index,
        right_finger_frame_index,
        wrist_frame_index,
    ]
    positions_w = frames.data.target_pos_w[:, indices]
    shelf_quat = shelf.data.root_quat_w.unsqueeze(1).expand(-1, len(indices), -1)
    positions_s = math_utils.quat_apply_inverse(
        shelf_quat.reshape(-1, 4),
        (positions_w - shelf.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(positions_w.shape)
    thresholds = torch.tensor(
        [finger_clearance, finger_clearance, wrist_clearance],
        dtype=positions_s.dtype,
        device=positions_s.device,
    )
    return shelf_clearance_kernel(
        positions_s,
        thresholds,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        surface_height=surface_height,
    )
