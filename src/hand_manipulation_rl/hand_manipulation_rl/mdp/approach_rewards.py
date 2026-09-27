"""Dense, safety-aware approach reward for the inherited sweep variant."""

from __future__ import annotations

import math

import torch


Tensor = torch.Tensor


def contact_path_potential(
    selected_surface_position_w: Tensor,
    initial_surface_position_w: Tensor,
    object_position_w: Tensor,
    initial_object_position_w: Tensor,
    command_direction_w: Tensor,
    command_distance_m: Tensor,
    target_height_w: Tensor,
    *,
    free_approach_distance_m: float | Tensor,
    precontact_potential_fraction: float = 0.35,
    cross_track_weight: float = 1.0,
    height_weight: float = 1.0,
) -> Tensor:
    """Return contact-coupled hand progress on the commanded path.

    Hand motion receives a fixed share of dense credit over the calibrated
    pre-contact distance. Beyond that frontier, the remaining share requires
    both continued hand travel and actual Cube progress toward the goal.
    Consequently a hand that passes beside a stationary Cube cannot collect
    the full shaping return, while real pushing remains rewarded.

    Cross-track motion and deviation from the calibrated contact height reduce
    the potential.  The result is normalized by the complete approach-plus-
    command distance and clipped to ``[-1, 1]``.
    """

    _require_matching_vectors(
        selected_surface_position_w,
        initial_surface_position_w,
        command_direction_w,
    )
    if object_position_w.shape != selected_surface_position_w.shape:
        raise ValueError("object_position_w must match the surface position shape")
    if initial_object_position_w.shape != selected_surface_position_w.shape:
        raise ValueError("initial_object_position_w must match the surface position shape")
    batch_shape = selected_surface_position_w.shape[:-1]
    if command_distance_m.shape != batch_shape or target_height_w.shape != batch_shape:
        raise ValueError("command distance and target height must match the batch shape")
    free_distance = torch.as_tensor(
        free_approach_distance_m,
        dtype=selected_surface_position_w.dtype,
        device=selected_surface_position_w.device,
    )
    try:
        free_distance = torch.broadcast_to(free_distance, batch_shape)
    except RuntimeError as exc:
        raise ValueError("free_approach_distance_m must broadcast to the batch shape") from exc
    if torch.any(free_distance <= 0.0) or torch.any(~torch.isfinite(free_distance)):
        raise ValueError("free_approach_distance_m must be finite and positive")
    if not 0.0 < precontact_potential_fraction < 1.0 or not math.isfinite(
        precontact_potential_fraction
    ):
        raise ValueError("precontact_potential_fraction must be finite and in (0, 1)")
    if cross_track_weight <= 0.0 or not math.isfinite(cross_track_weight):
        raise ValueError("cross_track_weight must be finite and positive")
    if height_weight <= 0.0 or not math.isfinite(height_weight):
        raise ValueError("height_weight must be finite and positive")
    if torch.any(command_distance_m <= 0.0) or torch.any(~torch.isfinite(command_distance_m)):
        raise ValueError("command_distance_m must be finite and positive")
    if torch.any(~torch.isfinite(target_height_w)):
        raise ValueError("target_height_w must be finite")

    planar_direction_norm = torch.linalg.vector_norm(
        command_direction_w[..., :2], dim=-1, keepdim=True
    )
    if torch.any(planar_direction_norm <= 1.0e-8):
        raise ValueError("command_direction_w must have a non-zero planar component")

    return _contact_path_potential_impl(
        selected_surface_position_w,
        initial_surface_position_w,
        object_position_w,
        initial_object_position_w,
        command_direction_w,
        command_distance_m,
        target_height_w,
        free_approach_distance_m=free_distance,
        precontact_potential_fraction=precontact_potential_fraction,
        cross_track_weight=cross_track_weight,
        height_weight=height_weight,
    )


def _contact_path_potential_impl(
    selected_surface_position_w: Tensor,
    initial_surface_position_w: Tensor,
    object_position_w: Tensor,
    initial_object_position_w: Tensor,
    command_direction_w: Tensor,
    command_distance_m: Tensor,
    target_height_w: Tensor,
    *,
    free_approach_distance_m: Tensor,
    precontact_potential_fraction: float,
    cross_track_weight: float,
    height_weight: float,
) -> Tensor:
    """Hot-path implementation after episode/config validation."""

    planar_direction_norm = torch.linalg.vector_norm(
        command_direction_w[..., :2], dim=-1, keepdim=True
    )
    direction_xy = command_direction_w[..., :2] / planar_direction_norm
    surface_displacement_xy = (
        selected_surface_position_w[..., :2] - initial_surface_position_w[..., :2]
    )
    hand_forward = torch.sum(surface_displacement_xy * direction_xy, dim=-1)
    cross_track_xy = surface_displacement_xy - hand_forward.unsqueeze(-1) * direction_xy
    cross_track_error = torch.linalg.vector_norm(cross_track_xy, dim=-1)

    object_displacement_xy = object_position_w[..., :2] - initial_object_position_w[..., :2]
    object_forward = torch.sum(object_displacement_xy * direction_xy, dim=-1)
    object_fraction = torch.clamp(object_forward / command_distance_m, min=0.0, max=1.0)

    approach_fraction = torch.clamp(
        hand_forward / free_approach_distance_m, min=-1.0, max=1.0
    )
    hand_push_fraction = torch.clamp(
        (hand_forward - free_approach_distance_m) / command_distance_m,
        min=0.0,
        max=1.0,
    )
    push_fraction = torch.minimum(hand_push_fraction, object_fraction)
    credited_potential = (
        precontact_potential_fraction * approach_fraction
        + (1.0 - precontact_potential_fraction) * push_fraction
    )

    # The safe descent is a path, not a separately collectible objective.
    # With zero forward progress the desired height remains the reset height;
    # it reaches the calibrated contact height only at the free frontier.
    descent_alpha = torch.clamp(
        hand_forward / free_approach_distance_m, min=0.0, max=1.0
    )
    desired_height = initial_surface_position_w[..., 2] + descent_alpha * (
        target_height_w - initial_surface_position_w[..., 2]
    )
    height_error = torch.abs(selected_surface_position_w[..., 2] - desired_height)
    normalization = free_approach_distance_m + command_distance_m
    potential = credited_potential - (
        cross_track_weight * cross_track_error + height_weight * height_error
    ) / normalization
    return torch.clamp(potential, min=-1.0, max=1.0)


def potential_progress_rate(
    current_potential: Tensor,
    previous_potential: Tensor,
    step_dt: float,
) -> Tensor:
    """Return signed potential progress per second.

    Isaac Lab's RewardManager multiplies this rate by ``step_dt``.  The
    integrated reward is therefore exactly ``current - previous``: approach
    is positive, no motion is zero, and retreat or descent is negative.
    """

    if current_potential.shape != previous_potential.shape:
        raise ValueError("current and previous potential must have identical shapes")
    if step_dt <= 0.0 or not math.isfinite(step_dt):
        raise ValueError("step_dt must be finite and positive")
    return (current_potential - previous_potential) / step_dt


def one_sided_height_safety_penalty(
    current_height_w: Tensor,
    initial_height_w: Tensor,
    *,
    tolerance_m: float,
    band_m: float,
) -> Tensor:
    """Quadratically penalize descending below the reset height in ``[-1, 0]``."""

    if current_height_w.shape != initial_height_w.shape:
        raise ValueError("current and initial height must have identical shapes")
    if tolerance_m < 0.0 or not math.isfinite(tolerance_m):
        raise ValueError("tolerance_m must be finite and non-negative")
    if band_m <= 0.0 or not math.isfinite(band_m):
        raise ValueError("band_m must be finite and positive")
    deficit = torch.clamp_min(initial_height_w - current_height_w - tolerance_m, 0.0)
    normalized = torch.clamp(deficit / band_m, max=1.0)
    return -(normalized * normalized)


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


def selected_surface_contact_path(env) -> Tensor:
    """Isaac Lab adapter for the v1 contact-coupled path potential."""

    target_height = env.selected_surface_approach_target_w()[:, 2]
    return _contact_path_potential_impl(
        env.selected_surface_control_point_w(),
        env.initial_selected_surface_position_w,
        env.scene["target_object"].data.root_pos_w,
        env.object_initial_pos_w,
        env.command_direction_w,
        env.command_distance,
        target_height,
        free_approach_distance_m=env.approach_free_distance_m(),
        precontact_potential_fraction=env.cfg.task.approach_precontact_potential_fraction,
        cross_track_weight=env.cfg.task.approach_path_cross_track_weight,
        height_weight=env.cfg.task.approach_path_height_weight,
    )


__all__ = [
    "contact_path_potential",
    "one_sided_height_safety_penalty",
    "potential_progress_rate",
    "safe_surface_approach_reward",
    "selected_surface_contact_path",
    "selected_surface_object_approach",
]
