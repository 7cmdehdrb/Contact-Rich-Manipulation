"""Revision command sampling and OBB placement, in metres on the same device."""

import math
import torch

from .bbox_scene_layout import LayoutConfig, footprint_from_bounds, generate_layout


def sample_revision_commands(bounds, generator, spawn_radius=.02, goal_distance=.15,
                             edge_clearance=.02, excluded_x_half_angle=math.pi / 6,
                             min_goal_distance=.05):
    """Central disk and feasible bounded ray; default range remains 5--15 cm."""
    if not 0 <= excluded_x_half_angle < math.pi / 2:
        raise ValueError('Goal direction exclusion must be in [0, pi/2)')
    if (not math.isfinite(min_goal_distance) or min_goal_distance <= 0
            or spawn_radius < 0 or goal_distance < min_goal_distance or edge_clearance < 0):
        raise ValueError('Invalid Revision command distances')
    device, dtype = bounds.device, bounds.dtype
    half = bounds[:, :, :2].abs().amax(1)
    safe = torch.minimum(torch.tensor([.20, .50], device=device) - half - edge_clearance,
                         torch.tensor([.16, .46], device=device))
    if bool((safe < spawn_radius).any()):
        raise ValueError('Target footprint leaves no safe central spawn disk')
    draws = torch.rand((len(bounds), 2), device=device, dtype=dtype, generator=generator)
    angle = 2 * math.pi * draws[:, 0]
    radius = spawn_radius * draws[:, 1].sqrt()
    target = radius[:, None] * torch.stack((angle.cos(), angle.sin()), -1)
    # Draw uniformly from the two allowed angular intervals without rejection.
    angles = torch.rand((len(bounds), 32), device=device, dtype=dtype, generator=generator)
    sides = (angles >= .5).to(dtype)
    angles = excluded_x_half_angle + (angles * 2 % 1) * (math.pi - 2 * excluded_x_half_angle) + sides * math.pi
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    room = (safe[:, None] - direction.sign() * target[:, None]) / direction.abs().clamp_min(1e-12)
    distance = room.amin(-1).clamp_max(goal_distance)
    feasible = distance >= min_goal_distance
    # Central disk has a guaranteed feasible +/-Y fallback for supported sizes.
    fallback = torch.zeros((len(bounds), 1, 2), device=device, dtype=dtype)
    fallback[:, 0, 1] = torch.where(target[:, 1] > 0, -1., 1.)
    fallback_distance = (safe[:, 1] + target[:, 1].abs()).clamp_max(goal_distance)
    direction = torch.cat((direction, fallback), 1)
    distance = torch.cat((distance, fallback_distance[:, None]), 1)
    feasible = torch.cat((feasible, fallback_distance[:, None] >= min_goal_distance), 1)
    if not bool(feasible.any(-1).all()):
        raise ValueError(f'No valid goal ray of at least {min_goal_distance} metres')
    index = feasible.to(torch.int64).argmax(-1)
    rows = torch.arange(len(bounds), device=device)
    goal = target + direction[rows, index] * distance[rows, index, None]
    return target, goal


def revision_layout(bounds, generator, gray=1, white=1, **command_params):
    """Return shelf-local root poses; inactive slots have separate gravity-free parking."""
    if bounds.shape[1:] != (10, 2, 3):
        raise ValueError('Revision requires exactly ten scaled object bounds')
    target, goal = sample_revision_commands(bounds[:, 0], generator, **command_params)
    count, device = len(bounds), bounds.device
    target_yaw = torch.full((count,), -math.pi / 2, device=device)
    target_half, _ = footprint_from_bounds(bounds[:, 0], target_yaw)
    center_offset = (bounds[:, 0, 0, :2] + bounds[:, 0, 1, :2]) / 2
    start_center = target + center_offset
    goal_center = goal + center_offset
    to_board = lambda xy: torch.stack((xy[:, 1], .20 - xy[:, 0]), -1)
    obstacle_half = (bounds[:, 1:, 1, :2] - bounds[:, 1:, 0, :2]) / 2
    cfg = LayoutConfig(gray_padding=.065, max_gray=4, max_white=5)
    requested = torch.tensor([gray, white], device=device).expand(count, -1)
    result = generate_layout(to_board(start_center), to_board(goal_center), target_half,
                             target_yaw, obstacle_half, cfg=cfg, requested=requested,
                             generator=generator)
    yaw = torch.cat((torch.zeros((count, 1), device=device), result.yaw + math.pi / 2), 1)
    _, offsets = footprint_from_bounds(bounds, yaw)
    centers = torch.stack((.20 - result.centers[..., 1], result.centers[..., 0]), -1)
    local = torch.zeros((count, 10, 3), device=device, dtype=bounds.dtype)
    local[:, 0, :2] = target
    local[:, 1:, :2] = centers - offsets[:, 1:]
    local[..., 2] = 1.05 - bounds[..., 0, 2]
    active = torch.cat((torch.ones((count, 1), dtype=torch.bool, device=device), result.active), 1)
    parking = torch.zeros_like(local)
    parking[..., 2] = 50. + 2. * torch.arange(10, device=device)
    local = torch.where(active[..., None], local, parking)
    yaw = torch.where(active, yaw, 0.)
    return local, yaw, active, goal - target, result
