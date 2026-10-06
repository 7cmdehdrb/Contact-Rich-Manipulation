"""Simulator-independent EEF height relative to the actual tabletop."""

from __future__ import annotations

import math

import torch

from .action_math import quaternion_conjugate, quaternion_rotate


def eef_height_guard(
    control_pos_w: torch.Tensor,
    table_pos_w: torch.Tensor,
    table_quat_w: torch.Tensor,
    *,
    table_thickness_m: float,
    soft_height_m: float,
    hard_height_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return above-top height, unweighted negative rate, and hard failure.

    Table local +Z defines height, so translating an environment or rotating
    its table does not change the result. The rate is ``-relu(h - soft)``;
    the RewardManager applies its weight and dt. Hard failure includes the
    upper equality boundary and invalid geometry. Invalid rows return zero
    height/rate with failure=True, keeping metrics and rewards finite.
    Caller applies its episode-live mask to rate and failure.
    """

    if not math.isfinite(table_thickness_m) or table_thickness_m <= 0:
        raise ValueError("table_thickness_m must be finite and positive")
    if not all(math.isfinite(value) for value in (soft_height_m, hard_height_m)) or soft_height_m >= hard_height_m:
        raise ValueError("Height thresholds must be finite with soft_height_m < hard_height_m")
    if control_pos_w.ndim != 2 or control_pos_w.shape[-1] != 3:
        raise ValueError("control_pos_w must have shape (N, 3)")
    count = control_pos_w.shape[0]
    if table_pos_w.shape != (count, 3) or table_quat_w.shape != (count, 4):
        raise ValueError("Table pose must have shapes (N, 3) and (N, 4)")
    if not all(value.dtype.is_floating_point for value in (control_pos_w, table_pos_w, table_quat_w)):
        raise ValueError("EEF and Table poses must use floating-point tensors")

    valid = torch.isfinite(control_pos_w).all(-1) & torch.isfinite(table_pos_w).all(-1)
    valid &= torch.isfinite(table_quat_w).all(-1)
    position = torch.nan_to_num(control_pos_w, nan=0., posinf=0., neginf=0.)
    table_position = torch.nan_to_num(table_pos_w, nan=0., posinf=0., neginf=0.)
    quaternion = torch.nan_to_num(table_quat_w, nan=0., posinf=0., neginf=0.)
    norm = quaternion.norm(dim=-1, keepdim=True)
    normalizable = torch.isfinite(norm) & (norm > 1.e-8)
    valid &= normalizable[:, 0]
    identity = quaternion.new_tensor((1., 0., 0., 0.)).expand_as(quaternion)
    quaternion = torch.where(normalizable, quaternion/norm.clamp_min(1.e-8), identity)
    local = quaternion_rotate(quaternion_conjugate(quaternion), position-table_position)
    height = local[:, 2]-table_thickness_m/2
    valid &= torch.isfinite(height)
    height = torch.where(valid, height, torch.zeros_like(height))
    penalty = torch.where(valid, -(height-soft_height_m).clamp_min(0.), torch.zeros_like(height))
    return height, penalty, (~valid) | (height >= hard_height_m)


__all__ = ["eef_height_guard"]
