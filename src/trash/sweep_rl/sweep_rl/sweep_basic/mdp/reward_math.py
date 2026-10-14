"""Pure tensor kernels shared by the single-stage sweep rewards."""

from __future__ import annotations

import torch


def gaussian_distance(distance: torch.Tensor, *, std: float) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("std must be positive.")
    return torch.exp(-torch.square(distance / std))


def signed_square_alignment_score(alignment: torch.Tensor) -> torch.Tensor:
    """Preserve the alignment sign while squaring its magnitude."""
    return torch.sign(alignment) * torch.square(alignment)


def remaining_horizon_failure_kernel(
    failure: torch.Tensor,
    episode_length: torch.Tensor,
    *,
    max_episode_length: int,
    minimum_steps: float,
) -> torch.Tensor:
    """Return remaining rollout steps only for terminal safety failures."""
    if failure.dtype != torch.bool:
        raise ValueError("failure must be a boolean tensor.")
    if failure.shape != episode_length.shape:
        raise ValueError("failure and episode_length must have the same shape.")
    if max_episode_length <= 0 or minimum_steps < 0.0:
        raise ValueError("Invalid episode horizon or minimum step charge.")
    remaining = torch.clamp(
        max_episode_length - episode_length, min=0
    ).to(torch.float32)
    charged = torch.clamp(remaining, min=minimum_steps)
    return failure.float() * charged


def pushing_target_kernel(
    goal_distance: torch.Tensor,
    motion_gate: torch.Tensor,
    target_y_speed: torch.Tensor,
    *,
    goal_threshold: float,
    distance_scale: float,
    velocity_reward_min: float,
    velocity_penalty_min: float,
) -> torch.Tensor:
    """Piecewise pushing score from endpoint distance, contact pose, and Y speed."""
    if motion_gate.dtype != torch.bool:
        raise ValueError("motion_gate must be a boolean tensor.")
    if not (
        goal_distance.shape == motion_gate.shape == target_y_speed.shape
    ):
        raise ValueError("Pushing tensors must have matching shapes.")
    if min(goal_threshold, distance_scale, velocity_reward_min) <= 0.0:
        raise ValueError("Push reward thresholds must be positive.")
    if velocity_penalty_min <= velocity_reward_min:
        raise ValueError("velocity_penalty_min must exceed velocity_reward_min.")
    velocity_reward = torch.where(
        target_y_speed >= velocity_penalty_min,
        -torch.full_like(target_y_speed, 0.5),
        torch.where(
            target_y_speed > velocity_reward_min,
            torch.full_like(target_y_speed, 0.5),
            torch.zeros_like(target_y_speed),
        ),
    )
    near_goal = 2.0 * torch.exp(-5.0 * goal_distance)
    sweeping = motion_gate.float() * (
        1.0 - goal_distance / distance_scale + velocity_reward
    )
    return torch.where(goal_distance < goal_threshold, near_goal, sweeping)


def shelf_clearance_kernel(
    frame_positions_s: torch.Tensor,
    clearance_thresholds: torch.Tensor,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    surface_height: float,
) -> torch.Tensor:
    """Return normalized top-surface clearance intrusion for each environment."""
    if frame_positions_s.ndim != 3 or frame_positions_s.shape[-1] != 3:
        raise ValueError("frame_positions_s must have shape (N, F, 3).")
    if clearance_thresholds.shape != frame_positions_s.shape[1:2]:
        raise ValueError("One clearance threshold is required per frame.")
    if torch.any(clearance_thresholds <= 0.0):
        raise ValueError("Clearance thresholds must be positive.")
    inside = (
        (frame_positions_s[..., 0] >= x_bounds[0])
        & (frame_positions_s[..., 0] <= x_bounds[1])
        & (frame_positions_s[..., 1] >= y_bounds[0])
        & (frame_positions_s[..., 1] <= y_bounds[1])
    )
    height = frame_positions_s[..., 2] - surface_height
    intrusion = torch.clamp(
        (clearance_thresholds.unsqueeze(0) - height)
        / clearance_thresholds.unsqueeze(0),
        0.0,
        1.0,
    )
    return torch.sum(intrusion * inside.float(), dim=-1)
