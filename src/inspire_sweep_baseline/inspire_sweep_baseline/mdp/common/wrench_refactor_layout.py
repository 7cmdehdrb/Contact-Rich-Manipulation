"""Continuous unbiased goal sampling and fixed-pool OBB placement (pure torch)."""
import math
import torch
from .bbox_scene_layout import LayoutConfig, footprint_from_bounds, generate_layout


def sample_commands(bounds, generator, spawn_radius=.02, edge_clearance=.02,
                    excluded_x_half_angle=math.pi / 6):
    """Uniform allowed angle, then uniform distance from zero to the safe edge.

    Shelf-local +/-X are 12/6 o'clock; +/-Y are the two lateral directions.
    No ray clipping to 15 cm, minimum push length or direction fallback.
    """
    if not math.isfinite(excluded_x_half_angle) or not 0 <= excluded_x_half_angle < math.pi / 2:
        raise ValueError('Invalid direction deadzone')
    if not math.isfinite(spawn_radius) or spawn_radius < 0 or not math.isfinite(edge_clearance) or edge_clearance < .005:
        raise ValueError('Invalid spawn disk or edge clearance')
    half = bounds[:, :, :2].abs().amax(1)
    safe = bounds.new_tensor([.20, .50]) - half - edge_clearance
    if bool((safe <= spawn_radius).any()):
        raise ValueError('Target footprint leaves no central spawn disk')
    draws = torch.rand((len(bounds), 4), device=bounds.device, dtype=bounds.dtype, generator=generator)
    angle = 2 * math.pi * draws[:, 0]
    target = spawn_radius * draws[:, 1:2].sqrt() * torch.stack((angle.cos(), angle.sin()), -1)
    side = (draws[:, 2] >= .5).to(bounds.dtype)
    theta = excluded_x_half_angle + (2 * draws[:, 2] % 1) * (math.pi - 2 * excluded_x_half_angle) + side * math.pi
    direction = torch.stack((theta.cos(), theta.sin()), -1)
    maximum = ((safe - direction.sign() * target) / direction.abs().clamp_min(1e-12)).amin(-1)
    distance = maximum * draws[:, 3]
    return target, target + direction * distance[:, None]


def fixed_pool_layout(bounds, spawn_z, generator, requested, *, red_margin=.023,
                      rear_offset=.02, **command_params):
    """Select six distinct candidate bodies from sixteen; never rebuild geometry.

    Only six candidates enter OBB search. Unused bodies retain distinct parking
    poses. Retry failed placements with the SAME goal/counts; never silently
    lower a sampled population or bias goals by resampling them.
    """
    count, device = len(bounds), bounds.device
    if bounds.shape[1:] != (17, 2, 3) or requested.shape != (count, 2):
        raise ValueError('Expected target + sixteen fixed bodies and two counts')
    if bool(((requested < 0) | (requested > 3)).any()):
        raise ValueError('Gray/White counts must be in [0,3]')
    target, goal = sample_commands(bounds[:, 0], generator, **command_params)
    selected = torch.rand((count, 16), device=device, generator=generator).argsort(-1)[:, :6] + 1
    rows = torch.arange(count, device=device)[:, None]
    chosen = bounds[rows, selected]
    target_yaw = bounds.new_full((count,), -math.pi / 2)
    target_half, _ = footprint_from_bounds(bounds[:, 0], target_yaw)
    offset = bounds[:, 0, :, :2].mean(1)
    to_board = lambda p: torch.stack((p[:, 1], .20 - p[:, 0]), -1)
    obstacle_half = (chosen[:, :, 1, :2] - chosen[:, :, 0, :2]) / 2
    cfg = LayoutConfig(gray_padding=.065, red_margin=red_margin, rear_offset=rear_offset,
                       max_gray=3, max_white=3, candidates=64, restarts=4, passes=2)
    result = generate_layout(to_board(target + offset), to_board(goal + offset), target_half,
                             target_yaw, obstacle_half, cfg=cfg, requested=requested, generator=generator)
    for _ in range(3):
        missing = (~result.complete).nonzero().flatten()
        if not len(missing):
            break
        retry = generate_layout(to_board(target[missing] + offset[missing]), to_board(goal[missing] + offset[missing]),
                                target_half[missing], target_yaw[missing], obstacle_half[missing],
                                cfg=cfg, requested=requested[missing], generator=generator)
        for name in ('centers', 'yaw', 'active', 'actual', 'complete'):
            getattr(result, name)[missing] = getattr(retry, name)
    if not bool(result.complete.all()):
        failed = (~result.complete).nonzero().flatten().tolist()
        raise RuntimeError(f'Fixed-pool OBB search exhausted for batch rows {failed}; goal/counts were not resampled')
    yaw = bounds.new_zeros((count, 17))
    yaw[rows, selected] = result.yaw + math.pi / 2
    _, offsets = footprint_from_bounds(chosen, result.yaw + math.pi / 2)
    centers = torch.stack((.20 - result.centers[..., 1], result.centers[..., 0]), -1)
    local = bounds.new_zeros((count, 17, 3))
    local[..., 2] = 50. + 2. * torch.arange(17, device=device)
    local[:, 0, :2], local[:, 0, 2] = target, spawn_z[:, 0]
    active = torch.zeros((count, 17), dtype=torch.bool, device=device)
    active[:, 0] = True
    active[rows, selected] = result.active
    candidate = torch.cat((centers - offsets, spawn_z[rows, selected, None]), -1)
    local[rows, selected] = torch.where(result.active[..., None], candidate, local[rows, selected])
    yaw = torch.where(active, yaw, 0.)
    return local, yaw, active, goal - target, result.actual
