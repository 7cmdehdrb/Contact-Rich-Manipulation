"""Reward and failure reductions over the episode's active object slots."""

import math

import torch

from ..contact_sweep_clutter.mdp import all_object_tilt_angles


def active_object_mask(env, object_values):
    """Return [env, object] membership, matching the supplied object's tensor.

    Older callers without variable object slots retain their all-active
    behavior. The reset event owns the mask; reward and failure queries never
    change membership during an episode.
    """
    mask = getattr(env, "_active_object_mask", None)
    shape = object_values.shape[:2]
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=object_values.device)
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError("_active_object_mask must be boolean [num_envs, num_objects]")
    return mask


def combined_active_object_cost(
    env, free_angle=math.radians(5.0), fail_angle=0.9,
    displacement_ratio=2.5, displacement_max_cost=0.04,
):
    """Retain the parent tilt/displacement scales, excluding inactive slots.

    Active targets still incur tilt cost; only active non-target objects incur
    displacement cost. Apply the displacement cap after summing active slots,
    exactly as in the parent. Parked states (including NaNs) do not contribute.
    """
    if not (0.0 <= free_angle < fail_angle <= math.pi):
        raise ValueError("Tilt thresholds must satisfy 0 <= free_angle < fail_angle <= pi")
    if not math.isfinite(displacement_max_cost) or displacement_max_cost <= 0.0:
        raise ValueError("displacement_max_cost must be finite and positive")
    angles = all_object_tilt_angles(env)
    active = active_object_mask(env, angles)
    per_object_tilt = ((angles - free_angle) / (fail_angle - free_angle)).clamp(0.0, 1.0).square()
    tilt = torch.where(active, per_object_tilt, 0.0).sum(dim=-1)
    position = env.scene["object_collection"].data.object_pos_w
    squared = (position - env._initial_object_positions_w).square().sum(dim=-1)
    object_ids = torch.arange(position.shape[1], device=position.device).view(1, -1)
    obstacles = active & (object_ids != env.target_id.reshape(env.num_envs, 1))
    displacement = torch.where(obstacles, squared, 0.0).sum(dim=-1)
    displacement = torch.nan_to_num(
        displacement, nan=displacement_max_cost, posinf=displacement_max_cost, neginf=displacement_max_cost,
    ).clamp(max=displacement_max_cost)
    return tilt + displacement_ratio * displacement


def any_active_object_fall_or_tilt(env, height_condition=1.04, fail_angle=0.9):
    """Apply the existing height/tilt rule to active targets and obstacles."""
    position = env.scene["object_collection"].data.object_pos_w
    height = position[..., 2] - env.scene.env_origins[:, None, 2]
    failed = (height < height_condition) | (all_object_tilt_angles(env) >= fail_angle)
    return (failed & active_object_mask(env, position)).any(dim=-1)
