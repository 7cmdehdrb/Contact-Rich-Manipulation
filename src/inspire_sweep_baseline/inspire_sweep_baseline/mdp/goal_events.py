"""Centered shelf spawns and goals with a uniformly sampled planar direction."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_mul


def _sample_uniform_disk(count, radius, device):
    radial_distance = radius * torch.rand(count, device=device).sqrt()
    angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
    return radial_distance[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)


def _sample_planar_unit_vectors(count, device):
    angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
    return torch.stack((angle.cos(), angle.sin()), dim=-1)


def _goal_ray_distance(spawn_xy, direction_xy, distance, safe_half_extents):
    """Keep the sampled direction and shorten its ray at the safe board edge."""
    half_extents = torch.as_tensor(safe_half_extents, device=spawn_xy.device, dtype=spawn_xy.dtype)
    limits = torch.where(direction_xy >= 0.0, half_extents, -half_extents)
    parallel = direction_xy == 0.0
    denominator = torch.where(parallel, torch.ones_like(direction_xy), direction_xy)
    room = (limits - spawn_xy) / denominator
    room = torch.where(parallel, torch.full_like(room, torch.inf), room)
    return room.amin(dim=-1).clamp(max=distance)


def randomize_goal_object_scene(
    env,
    env_ids,
    spawn_radius=0.02,
    goal_distance=0.15,
    object_radius=0.04,
    edge_clearance=0.02,
    shelf_half_extents=(0.20, 0.50),
    shelf_surface_height=1.05,
):
    """Reset one cylinder on the shelf and choose an episode-fixed XY goal.

    Sample uniformly by area from a disk centered on the shelf's active board.
    Independently sample every azimuth uniformly. The requested goal distance
    is shortened where necessary to retain the cylinder radius plus the
    additional edge clearance. The spawn radius and sampled direction stay
    unchanged. All geometry is expressed in the shelf's current local frame.
    """
    values = (spawn_radius, goal_distance, object_radius, edge_clearance,
              shelf_surface_height, *shelf_half_extents)
    if len(shelf_half_extents) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Shelf geometry and sampling parameters must be finite; shelf_half_extents must have two axes")
    if spawn_radius < 0.0 or goal_distance <= 0.0 or object_radius <= 0.0 or edge_clearance < 0.0:
        raise ValueError("spawn_radius and edge_clearance must be nonnegative; goal_distance and object_radius must be positive")
    safe_half_extents = tuple(half - object_radius - edge_clearance for half in shelf_half_extents)
    if min(safe_half_extents) <= 0.0 or spawn_radius > min(safe_half_extents):
        raise ValueError("The complete spawn disk must fit inside the shelf with object radius and edge clearance")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    count = env_ids.numel()
    if count == 0:
        return
    objects = env.scene["object_collection"]
    if objects.num_objects != 1:
        raise RuntimeError("The goal sweep requires exactly one cylinder in object_collection")

    spawn_xy = _sample_uniform_disk(count, spawn_radius, env.device)
    direction_xy = _sample_planar_unit_vectors(count, env.device)
    distance = _goal_ray_distance(spawn_xy, direction_xy, goal_distance, safe_half_extents)
    spawn_local = torch.cat((spawn_xy, torch.full((count, 1), shelf_surface_height, device=env.device)), dim=-1)
    displacement_local = torch.cat((direction_xy * distance[:, None], torch.zeros((count, 1), device=env.device)), dim=-1)
    shelf = env.scene["shelf"].data
    shelf_quat = shelf.root_quat_w[env_ids]
    positions = shelf.root_pos_w[env_ids] + quat_apply(shelf_quat, spawn_local)
    displacement_world = quat_apply(shelf_quat, displacement_local)

    yaw = torch.empty(count, device=env.device).uniform_(-math.pi, math.pi)
    yaw_quat = torch.zeros((count, 4), device=env.device)
    yaw_quat[:, 0] = (yaw / 2.0).cos()
    yaw_quat[:, 3] = (yaw / 2.0).sin()
    rotations = quat_mul(shelf_quat, yaw_quat)
    states = torch.cat((positions, rotations, torch.zeros((count, 6), device=env.device)), dim=-1)
    objects.write_object_link_state_to_sim(
        states.unsqueeze(1), env_ids=env_ids,
        object_ids=torch.tensor([0], device=env.device, dtype=torch.long),
    )
    env.target_id[env_ids, 0] = 0
    env.target_width[env_ids, 0] = object_radius
    # SweepGoalCommand snapshots the reset pose and adds this world-frame vector.
    env.sweep_dir[env_ids] = displacement_world
