"""Pure tensor geometry for bounded, accumulated Cartesian position targets."""

from __future__ import annotations

import math

import torch

from .action_math import quaternion_conjugate, quaternion_rotate


def accumulated_translation_target(
    previous_target_b: torch.Tensor,
    current_position_b: torch.Tensor,
    current_quaternion_b: torch.Tensor,
    delta_position_c: torch.Tensor,
    error_limit_m: float,
) -> torch.Tensor:
    """Integrate a current-C increment and bound target error at this update.

    The target persists in the robot base frame. Rotating the hand changes the
    axes of subsequent increments, without rotating the previously accumulated
    target. Projection onto an error ball prevents unbounded position windup.
    """

    if not math.isfinite(error_limit_m) or error_limit_m <= 0.0:
        raise ValueError("Position target error limit must be finite and positive")
    proposed = previous_target_b + quaternion_rotate(current_quaternion_b, delta_position_c)
    error = proposed - current_position_b
    norm = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
    scale = error_limit_m / norm.clamp_min(error_limit_m)
    return current_position_b + error * scale


def translation_target_error_c(
    target_position_b: torch.Tensor,
    current_position_b: torch.Tensor,
    current_quaternion_b: torch.Tensor,
) -> torch.Tensor:
    """Express the persistent position target error in the current C axes."""

    return quaternion_rotate(
        quaternion_conjugate(current_quaternion_b), target_position_b - current_position_b
    )


__all__ = ["accumulated_translation_target", "translation_target_error_c"]
