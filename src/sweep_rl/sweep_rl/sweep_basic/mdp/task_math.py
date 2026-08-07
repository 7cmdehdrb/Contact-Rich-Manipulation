"""Pure tensor kernels for sweep success and failure conditions."""

from __future__ import annotations

import torch


def target_reached_and_stopped_mask(
    goal_distance: torch.Tensor,
    object_speed: torch.Tensor,
    *,
    distance_threshold: float,
    speed_threshold: float,
) -> torch.Tensor:
    if distance_threshold <= 0.0 or speed_threshold <= 0.0:
        raise ValueError("Invalid sweep success thresholds.")
    return (goal_distance < distance_threshold) & (object_speed < speed_threshold)


def update_contact_entry_state(
    contact_elapsed: torch.Tensor,
    entered: torch.Tensor,
    contact: torch.Tensor,
    *,
    dt: float,
    dwell_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate continuous real contact and latch completed entry."""
    if contact.dtype != torch.bool or entered.dtype != torch.bool:
        raise ValueError("contact and entered must be boolean tensors.")
    if not (contact_elapsed.shape == entered.shape == contact.shape):
        raise ValueError("Entry state tensors must have the same shape.")
    if dt <= 0.0 or dwell_time <= 0.0:
        raise ValueError("dt and dwell_time must be positive.")
    next_elapsed = torch.where(
        contact,
        contact_elapsed + dt,
        torch.zeros_like(contact_elapsed),
    )
    next_entered = entered | (next_elapsed + 1.0e-6 >= dwell_time)
    return next_elapsed, next_entered


def midpoint_no_entry_mask(
    episode_length: torch.Tensor,
    entered: torch.Tensor,
    *,
    max_episode_length: int,
    episode_fraction: float,
) -> torch.Tensor:
    """Return environments missing entry after the chosen horizon fraction."""
    if entered.dtype != torch.bool or entered.shape != episode_length.shape:
        raise ValueError("entered must be boolean and match episode_length.")
    if max_episode_length <= 0 or not 0.0 < episode_fraction < 1.0:
        raise ValueError("Invalid episode horizon or fraction.")
    reached_midpoint = (
        episode_length.to(torch.float32)
        >= max_episode_length * episode_fraction
    )
    return reached_midpoint & ~entered


def object_drop_mask(
    current_height: torch.Tensor,
    initial_height: torch.Tensor,
    *,
    maximum_height_change: float,
) -> torch.Tensor:
    if maximum_height_change <= 0.0:
        raise ValueError("maximum_height_change must be positive.")
    return torch.abs(current_height - initial_height) > maximum_height_change


def speed_limit_mask(velocity: torch.Tensor, *, maximum_speed: float) -> torch.Tensor:
    if maximum_speed <= 0.0:
        raise ValueError("maximum_speed must be positive.")
    return torch.linalg.norm(velocity, dim=-1) > maximum_speed


def contact_force_mask(
    force_matrix_w: torch.Tensor,
    *,
    force_threshold: float,
    fail_on_nonfinite: bool = True,
) -> torch.Tensor:
    """Detect filtered contact force or an invalid contact tensor per environment."""
    if force_threshold <= 0.0:
        raise ValueError("force_threshold must be positive.")
    if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
        raise ValueError("force_matrix_w must have shape (N, B, F, 3).")
    finite = torch.isfinite(force_matrix_w).all(dim=(1, 2, 3))
    safe_force = torch.nan_to_num(
        force_matrix_w,
        nan=0.0,
        posinf=force_threshold + 1.0,
        neginf=-(force_threshold + 1.0),
    )
    force = torch.linalg.norm(safe_force, dim=-1)
    contact = torch.any(force > force_threshold, dim=(1, 2))
    return (contact | ~finite) if fail_on_nonfinite else (contact & finite)


def update_sustained_violation(
    consecutive_steps: torch.Tensor,
    violation: torch.Tensor,
    *,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count consecutive violations and return the sustained failure mask."""
    if required_steps <= 0:
        raise ValueError("required_steps must be positive.")
    next_steps = torch.where(
        violation, consecutive_steps + 1, torch.zeros_like(consecutive_steps)
    )
    return next_steps, next_steps >= required_steps
