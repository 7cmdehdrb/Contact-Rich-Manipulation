"""Revision observations: measured joints, initial Actor, and live Critic geometry."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


OBSTACLE_OBSERVATION_SLOTS = 16


def arm_joint_positions(env, asset_cfg: SceneEntityCfg):
    """Actual arm angles in radians, without subtracting the default pose."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def arm_joint_velocities(env, asset_cfg: SceneEntityCfg):
    """Actual arm angular velocities, without subtracting default velocities."""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def pack_obstacle_rows(rows, adjacent_swap_probability=0.1):
    """Sort reported root Y descending, then try each occupied adjacent pair.

    Trials are independent; later pairs see earlier swaps. Unreported rows are
    exactly zero and remain a suffix. Actor calls this once at reset; Critic
    calls it for every observation, after reading current physical positions.
    """
    if not math.isfinite(adjacent_swap_probability) or not 0 <= adjacent_swap_probability <= 1:
        raise ValueError("adjacent_swap_probability must be in [0, 1]")
    present = rows[..., 0] > 0.5
    keys = torch.where(present, -rows[..., 2], math.inf)
    order = keys.argsort(dim=1, stable=True)
    packed = torch.where(present[..., None], rows, 0.0).gather(
        1, order[..., None].expand(-1, -1, 4),
    )
    if adjacent_swap_probability == 0:
        return packed
    trials = torch.rand((rows.shape[0], max(0, rows.shape[1] - 1)), device=rows.device)
    for index in range(rows.shape[1] - 1):
        swap = ((packed[:, index, 0] > 0.5) & (packed[:, index + 1, 0] > 0.5)
                & (trials[:, index] < adjacent_swap_probability))
        left, right = packed[:, index].clone(), packed[:, index + 1].clone()
        packed[:, index] = torch.where(swap[:, None], right, left)
        packed[:, index + 1] = torch.where(swap[:, None], left, right)
    return packed


def make_obstacle_observation(
    positions_root, active, *, visible=None, visibility_flip_probability=0.0,
    coordinate_noise=0.0, adjacent_swap_probability=0.1,
    observation_slots=OBSTACLE_OBSERVATION_SLOTS,
):
    """Pack actual obstacle roles into fixed-capacity ``[mask, x, y, z]`` rows.

    The caller excludes the target from ``active``. Visibility mistakes can
    affect only actual placed obstacles: unused physical slots cannot become
    observations, even when every visibility bit is flipped. Coordinate noise
    is uniform +/- the supplied magnitude; Critic supplies zero and no filter.
    """
    if positions_root.ndim != 3 or positions_root.shape[-1] != 3:
        raise ValueError("positions_root must have shape (num_envs, num_objects, 3)")
    if active.shape != positions_root.shape[:-1]:
        raise ValueError("active must identify each physical object")
    if not math.isfinite(visibility_flip_probability) or not 0 <= visibility_flip_probability <= 1:
        raise ValueError("visibility_flip_probability must be in [0, 1]")
    if not math.isfinite(coordinate_noise) or coordinate_noise < 0:
        raise ValueError("coordinate_noise must be finite and nonnegative")
    if observation_slots <= 0:
        raise ValueError("observation_slots must be positive")
    active = active.bool()
    if positions_root.shape[1] > observation_slots and bool((active.sum(-1) > observation_slots).any()):
        raise ValueError("Actual obstacles exceed the observation slot capacity")
    reported = active if visible is None else visible.bool() & active
    if reported.shape != active.shape:
        raise ValueError("visible must identify each physical object")
    if visibility_flip_probability:
        flips = torch.rand(reported.shape, device=reported.device) < visibility_flip_probability
        reported = torch.logical_xor(reported, flips) & active
    positions = positions_root
    if coordinate_noise:
        positions = positions + (2.0 * torch.rand_like(positions) - 1.0) * coordinate_noise
    rows = torch.cat((reported[..., None].to(positions.dtype), positions), dim=-1)
    rows = pack_obstacle_rows(rows, adjacent_swap_probability)
    if rows.shape[1] < observation_slots:
        rows = torch.cat((rows, rows.new_zeros((rows.shape[0], observation_slots - rows.shape[1], 4))), dim=1)
    return rows[:, :observation_slots].flatten(1)


def initial_obstacle_positions(env):
    """Read the 64D reset-time snapshot; never refresh it from live geometry."""
    rows = env._initial_obstacle_observation.unflatten(-1, (OBSTACLE_OBSERVATION_SLOTS, 4))
    # Return an owned tensor so observation modifiers cannot mutate the cache.
    return torch.where(rows[..., :1] > 0.5, rows, 0.0).flatten(1)


def current_obstacle_positions(env, adjacent_swap_probability=0.1):
    """Live positions of all actual obstacle roles, without noise or occlusion.

    Activity comes from the placement result, never from proximity to the shelf.
    An active obstacle moved or knocked off the shelf remains observable until
    reset. Target identity is excluded even when physical roles are reassigned.
    """
    objects = env.scene["object_collection"]
    robot = env.scene["robot"].data
    positions = objects.data.object_pos_w
    active = env._active_object_mask.clone().bool()
    active.scatter_(1, env.target_id.reshape(-1, 1).long(), False)
    roots = robot.root_quat_w[:, None].expand(-1, positions.shape[1], -1)
    positions_root = quat_apply_inverse(roots, positions - robot.root_pos_w[:, None])
    return make_obstacle_observation(
        positions_root, active, adjacent_swap_probability=adjacent_swap_probability,
    )
