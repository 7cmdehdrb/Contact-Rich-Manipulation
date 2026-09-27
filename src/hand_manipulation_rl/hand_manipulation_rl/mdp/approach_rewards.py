"""Dense, safety-aware approach reward for the inherited sweep variant."""

from __future__ import annotations

import math

import torch


Tensor = torch.Tensor


def _require_matching_vectors(
    selected_surface_position_w: Tensor,
    object_position_w: Tensor,
    command_direction_w: Tensor,
) -> None:
    expected = selected_surface_position_w.shape
    if selected_surface_position_w.ndim < 1 or expected[-1] != 3:
        raise ValueError("selected_surface_position_w must have shape (..., 3)")
    if object_position_w.shape != expected or command_direction_w.shape != expected:
        raise ValueError("approach positions and command directions must have identical (..., 3) shapes")


def safe_surface_approach_reward(
    selected_surface_position_w: Tensor,
    object_position_w: Tensor,
    command_direction_w: Tensor,
    safe_height_offset_m: float | Tensor,
    *,
    object_half_extent_m: float,
    planar_capture_radius_m: float,
    planar_sigma_m: float,
    height_sigma_m: float,
) -> Tensor:
    """Reward approaching the upstream object face at a board-safe height.

    The horizontal target is the object face opposite the commanded sweep.
    A small one-sided capture radius makes the score flat immediately outside
    that face, while penetration always reduces it.  The independent height
    factor is maximal only at the collision-safe reset height, so descending
    toward the board never improves this term.
    """

    _require_matching_vectors(
        selected_surface_position_w,
        object_position_w,
        command_direction_w,
    )
    if object_half_extent_m <= 0.0 or not math.isfinite(object_half_extent_m):
        raise ValueError("object_half_extent_m must be finite and positive")
    if planar_capture_radius_m < 0.0 or not math.isfinite(planar_capture_radius_m):
        raise ValueError("planar_capture_radius_m must be finite and non-negative")
    if planar_sigma_m <= 0.0 or not math.isfinite(planar_sigma_m):
        raise ValueError("planar_sigma_m must be finite and positive")
    if height_sigma_m <= 0.0 or not math.isfinite(height_sigma_m):
        raise ValueError("height_sigma_m must be finite and positive")

    planar_direction_norm = torch.linalg.vector_norm(
        command_direction_w[..., :2], dim=-1, keepdim=True
    )
    if torch.any(planar_direction_norm <= 1.0e-8):
        raise ValueError("command_direction_w must have a non-zero planar component")

    safe_height = torch.as_tensor(
        safe_height_offset_m,
        dtype=selected_surface_position_w.dtype,
        device=selected_surface_position_w.device,
    )
    if torch.any(~torch.isfinite(safe_height)):
        raise ValueError("safe_height_offset_m must be finite")
    try:
        safe_height = torch.broadcast_to(safe_height, selected_surface_position_w.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("safe_height_offset_m must broadcast to the batch shape") from exc

    return _safe_surface_approach_reward_impl(
        selected_surface_position_w,
        object_position_w,
        command_direction_w,
        safe_height,
        object_half_extent_m=object_half_extent_m,
        planar_capture_radius_m=planar_capture_radius_m,
        planar_sigma_m=planar_sigma_m,
        height_sigma_m=height_sigma_m,
    )


def _safe_surface_approach_reward_impl(
    selected_surface_position_w: Tensor,
    object_position_w: Tensor,
    command_direction_w: Tensor,
    safe_height_offset_m: Tensor,
    *,
    object_half_extent_m: float,
    planar_capture_radius_m: float,
    planar_sigma_m: float,
    height_sigma_m: float,
) -> Tensor:
    """Hot-path implementation after static inputs have been validated."""

    direction_xy = command_direction_w[..., :2]
    direction_xy = direction_xy / torch.clamp_min(
        torch.linalg.vector_norm(direction_xy, dim=-1, keepdim=True), 1.0e-8
    )
    target_position_xy = (
        object_position_w[..., :2] - object_half_extent_m * direction_xy
    )
    relative_xy = selected_surface_position_w[..., :2] - target_position_xy
    planar_distance = torch.linalg.vector_norm(relative_xy, dim=-1)

    # The capture zone is one-sided: a small region immediately outside the
    # upstream face is flat, while crossing into the Cube is always worse.
    capture_error = torch.clamp_min(
        planar_distance - planar_capture_radius_m, 0.0
    )
    penetration_error = torch.clamp_min(
        torch.sum(relative_xy * direction_xy, dim=-1), 0.0
    )
    planar_error = capture_error + penetration_error
    target_height_w = object_position_w[..., 2] + safe_height_offset_m
    height_error = torch.abs(selected_surface_position_w[..., 2] - target_height_w)
    return torch.exp(-planar_error / planar_sigma_m - height_error / height_sigma_m)


def selected_surface_object_approach(env) -> Tensor:
    """Isaac Lab adapter for :func:`safe_surface_approach_reward`."""

    # Configuration values are validated at construction and this task emits
    # exact planar commands at reset. Calling the implementation directly
    # avoids a pair of device-to-host synchronizations in every GPU reward step.
    return _safe_surface_approach_reward_impl(
        env.selected_surface_control_point_w(),
        env.scene["target_object"].data.root_pos_w,
        env.command_direction_w,
        env.safe_approach_height_offset_m(),
        object_half_extent_m=0.5 * env.cfg.task.cube_size,
        planar_capture_radius_m=env.cfg.task.approach_planar_capture_radius_m,
        planar_sigma_m=env.cfg.task.approach_planar_sigma_m,
        height_sigma_m=env.cfg.task.approach_height_sigma_m,
    )


__all__ = ["safe_surface_approach_reward", "selected_surface_object_approach"]
