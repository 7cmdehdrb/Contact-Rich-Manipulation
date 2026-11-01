"""Revision goal geometry and shared constants, independent of asset root origins."""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT

GOAL_SUCCESS_RADIUS = 0.03
HOLD_OUTER_RADIUS = 0.08
# Eight normalized base shapes average 0.118154574 m high (USD bounds).
# Round to a fixed 0.120 m above the initial surface, never the live target top.
CONTACT_HEIGHT_LIMIT = 0.12


def goal_radius(env):
    """Read the one Revision radius used by success, metrics and visualization."""
    radius = float(env.cfg.goal_success_radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Revision success radius must be finite and positive")
    return radius


def object_geometry_in_shelf(env):
    """Return root positions and lowest bbox points in the current shelf frame.

    The authoritative bounds already contain the instance scale. Transforming
    all eight corners handles asymmetric asset origins and current object tilt.
    """
    data = env.scene["object_collection"].data
    shelf = env.scene["shelf"].data
    count = data.object_pos_w.shape[1]
    shelf_q = shelf.root_quat_w[:, None].expand(-1, count, -1)
    root = quat_apply_inverse(shelf_q, data.object_pos_w - shelf.root_pos_w[:, None])
    rotation = quat_mul(quat_conjugate(shelf_q), data.object_quat_w)
    bounds = env._revision_bounds_t
    if bounds.shape != (*data.object_pos_w.shape[:2], 2, 3):
        raise ValueError("Revision bounds must be [num_envs, num_objects, 2, 3]")
    choices = torch.tensor(
        ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
         (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)),
        device=bounds.device, dtype=torch.bool,
    )
    corners = torch.where(choices[None, None], bounds[:, :, None, 1], bounds[:, :, None, 0])
    corners = quat_apply(rotation[:, :, None].expand(-1, -1, 8, -1), corners)
    bottom = root[..., 2] + corners[..., 2].amin(dim=-1)
    return root, bottom


def target_measurements(env, command_name="target_goal_pos"):
    """Preserve v0_2 settling validity while checking the physical object bottom."""
    data = env.scene["object_collection"].data
    rows = torch.arange(env.num_envs, device=env.device)
    ids = env.target_id[:, 0].long()
    position = data.object_pos_w[rows, ids]
    velocity = data.object_lin_vel_w[rows, ids]
    goal = env.command_manager.get_command(command_name)
    local, bottom = object_geometry_in_shelf(env)
    up = torch.zeros_like(position)
    up[:, 2] = 1.0
    alignment = (
        quat_apply(data.object_quat_w[rows, ids], up)
        * quat_apply(env.scene["shelf"].data.root_quat_w, up)
    ).sum(-1)
    valid = (
        torch.isfinite(position).all(-1)
        & torch.isfinite(velocity).all(-1)
        & torch.isfinite(goal[:, :2]).all(-1)
        & torch.isfinite(bottom[rows, ids])
        & torch.isfinite(alignment)
        & env._active_object_mask[rows, ids]
        & (local[rows, ids, 0].abs() <= 0.16)
        & (local[rows, ids, 1].abs() <= 0.46)
        & (bottom[rows, ids] >= SHELF_SURFACE_HEIGHT - 0.01)
        & (bottom[rows, ids] <= SHELF_SURFACE_HEIGHT + 0.02)
        & (alignment >= math.cos(0.9))
    )
    distance = torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    return distance, speed, valid
