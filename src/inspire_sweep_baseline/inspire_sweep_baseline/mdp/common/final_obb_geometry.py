"""Tensor OBB layout adapter for Final's shelf frame and fixed logical slots."""
import math
import torch
from .bbox_scene_layout import LayoutConfig, axes, generate_layout
from .contact_sweep_clutter_fusion_population import PARKING_HEIGHT


def sample_commands(bounds, generator, spawn_radius=.02, goal_distance=.15,
                    edge_clearance=.02, shelf_half_extents=(.20, .50),
                    excluded_x_half_angle=math.radians(30), wide_target=False):
    """Sample root-frame commands with exact local bounds, preserving Final's cone.

    Target yaw stays zero. The old circle-derived safe domain is replaced by
    the intersection of actual bounds, shelf margins and goal root limits.
    """
    if not 0 <= excluded_x_half_angle < math.pi / 2 or spawn_radius < 0 or goal_distance < .05:
        raise ValueError('Invalid Final command sampling parameters')
    n, device, dtype = len(bounds), bounds.device, bounds.dtype
    shelf = bounds.new_tensor(shelf_half_extents)
    low = torch.maximum(-shelf + edge_clearance - bounds[:, 0, :2], bounds.new_tensor([-.16, -.46]))
    high = torch.minimum(shelf - edge_clearance - bounds[:, 1, :2], bounds.new_tensor([.16, .46]))
    if wide_target:
        low = torch.maximum(low, -.8 * shelf)
        high = torch.minimum(high, .8 * shelf)
        target = low + torch.rand(n, 2, device=device, dtype=dtype, generator=generator) * (high-low)
        target_valid = (high >= low).all(-1)
    else:
        draws = torch.rand(n, 32, 2, device=device, dtype=dtype, generator=generator)
        angle = draws[..., 0] * (2 * math.pi)
        radius = draws[..., 1].sqrt() * spawn_radius
        candidates = radius[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
        valid = ((candidates >= low[:, None]) & (candidates <= high[:, None])).all(-1)
        target = candidates[torch.arange(n, device=device), valid.int().argmax(-1)]
        target_valid = valid.any(-1)
    angle = torch.rand(n, 32, device=device, dtype=dtype, generator=generator) * (2 * math.pi)
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    # A guaranteed axis fallback if random directions miss a feasible ray.
    fallback = torch.zeros(n, 1, 2, device=device, dtype=dtype)
    fallback[:, 0, 1] = torch.where(high[:, 1]-target[:, 1] >= target[:, 1]-low[:, 1], 1., -1.)
    direction = torch.cat((direction, fallback), dim=1)
    room = torch.where(direction > 0, high[:, None]-target[:, None], target[:, None]-low[:, None])
    room = torch.where(direction.abs() > 1e-7, room / direction.abs().clamp_min(1e-7), torch.inf)
    distance = room.amin(-1).clamp_max(goal_distance)
    valid = (distance >= .05) & (direction[..., 1].abs() >= math.sin(excluded_x_half_angle))
    if not (target_valid & valid.any(-1)).all():
        raise ValueError('No feasible target/goal for the selected bounds and command settings')
    selected = valid.int().argmax(-1)
    rows = torch.arange(n, device=device)
    return target, target + direction[rows, selected] * distance[rows, selected, None]


@torch.no_grad()
def sample_shelf_layout(bounds, generator, *, max_gray_slots=10, max_white_slots=10,
                        gray_obstacles=1, white_obstacles=1, gray_padding=.06, finger_gap=.02,
                        gray_red_padding_fraction=0., placement_candidates=32, placement_restarts=4,
                        placement_passes=2, spawn_radius=.02, goal_distance=.15, edge_clearance=.02,
                        shelf_half_extents=(.20, .50), shelf_surface_height=1.05,
                        excluded_x_half_angle=math.radians(30), wide_target=False):
    """Return root positions/yaws, sparse active mask, root displacement and counts.

    Physical/observation capacity is unchanged. Search only the 4 Gray + 5 White
    eligible slots; a failed slot remains inactive, without compacting identities.
    Legacy padding curriculum now expands Gray OUTWARD, never shrinks RED.
    """
    if not 0 <= gray_red_padding_fraction <= .15:
        raise ValueError('Gray padding fraction must be in [0,.15]')
    n, capacity = len(bounds), 1 + max_gray_slots + max_white_slots
    if bounds.shape != (n, capacity, 2, 3):
        raise ValueError('Expected scaled local bounds for each logical object')
    target, goal = sample_commands(bounds[:, 0], generator, spawn_radius, goal_distance,
        edge_clearance, shelf_half_extents, excluded_x_half_angle, wide_target)
    half = (bounds[..., 1, :2] - bounds[..., 0, :2]) / 2
    offset = (bounds[..., 1, :2] + bounds[..., 0, :2]) / 2
    gray_capacity, white_capacity = min(4, max_gray_slots), min(5, max_white_slots)
    indices = torch.tensor([*range(1, 1+gray_capacity),
                            *range(1+max_gray_slots, 1+max_gray_slots+white_capacity)], device=bounds.device)
    to_board = lambda p: torch.stack((p[..., 1], shelf_half_extents[0]-p[..., 0]), -1)
    target_center = target + offset[:, 0]
    cfg = LayoutConfig(board=(-shelf_half_extents[1], shelf_half_extents[1], 0., 2*shelf_half_extents[0]),
                       gray_padding=gray_padding*(1+gray_red_padding_fraction), finger_gap=finger_gap,
                       max_gray=gray_capacity, max_white=white_capacity, candidates=placement_candidates,
                       restarts=placement_restarts, passes=placement_passes)
    requested = torch.tensor([gray_obstacles, white_obstacles], device=bounds.device, dtype=torch.long).expand(n, -1)
    result = generate_layout(to_board(target_center), to_board(goal+offset[:, 0]), half[:, 0],
        bounds.new_full((n,), -math.pi/2), half[:, indices], cfg=cfg, requested=requested,
        generator=generator, check_inputs=False)
    yaw = bounds.new_zeros(n, capacity)
    yaw[:, indices] = result.yaw + math.pi/2
    rotated_offset = torch.einsum('bmi,bmij->bmj', offset[:, indices], axes(yaw[:, indices]))
    root_xy = torch.stack((shelf_half_extents[0]-result.centers[..., 1], result.centers[..., 0]), -1) - rotated_offset
    local = bounds.new_zeros(n, capacity, 3)
    local[..., 1] = torch.arange(capacity, device=bounds.device) * .2
    local[..., 2] = PARKING_HEIGHT
    local[:, 0, :2] = target
    local[:, 0, 2] = shelf_surface_height-bounds[:, 0, 0, 2]
    local[:, indices, :2] = torch.where(result.active[..., None], root_xy, local[:, indices, :2])
    local[:, indices, 2] = torch.where(result.active, shelf_surface_height-bounds[:, indices, 0, 2], local[:, indices, 2])
    active = torch.zeros(n, capacity, device=bounds.device, dtype=torch.bool)
    active[:, 0] = True
    active[:, indices] = result.active
    return local, yaw, active, goal-target, result.actual
