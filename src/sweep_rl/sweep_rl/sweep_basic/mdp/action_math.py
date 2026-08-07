"""Pure tensor helpers for the sweep arm action."""

from __future__ import annotations

import torch


def incremental_joint_target(
    current_joint_pos: torch.Tensor,
    action: torch.Tensor,
    joint_limits: torch.Tensor,
    *,
    scale: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized actions into one-step, current-relative joint targets.

    The returned target is latched by the action term for the whole control
    interval.  This avoids accumulating the same delta once per physics
    substep while still removing the old ``q_default +/- scale`` workspace
    restriction.
    """
    if current_joint_pos.shape != action.shape:
        raise ValueError("current_joint_pos and action must have the same shape.")
    if joint_limits.shape != (*current_joint_pos.shape, 2):
        raise ValueError("joint_limits must have shape (*joint_pos.shape, 2).")
    if isinstance(scale, (float, int)) and scale <= 0.0:
        raise ValueError("scale must be positive.")
    if isinstance(scale, torch.Tensor) and torch.any(scale <= 0.0):
        raise ValueError("scale must be positive.")

    normalized_action = torch.clamp(
        torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0),
        -1.0,
        1.0,
    )
    target = current_joint_pos + normalized_action * scale
    target = torch.clamp(
        target,
        min=joint_limits[..., 0],
        max=joint_limits[..., 1],
    )
    return normalized_action, target
