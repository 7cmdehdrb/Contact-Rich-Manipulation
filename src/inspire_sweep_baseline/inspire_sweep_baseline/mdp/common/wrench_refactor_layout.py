"""Continuous unbiased goal sampling and fixed-pool OBB placement (pure torch)."""
import math
import torch
from .bbox_scene_layout import LayoutConfig, footprint_from_bounds, generate_layout


def sample_commands(bounds, generator, spawn_radius=.02, edge_clearance=.02,
                    excluded_x_half_angle=math.pi / 6, min_goal_distance=.06, max_goal_distance=.12,
                    sweep_direction='both', goal_angle_range=None, fixed_goal_offset=None):
    """Uniform allowed angle, then uniform distance within the annulus and safe shelf edge.

    Shelf-local +X is robot-side (6 o'clock), -X is 12 o'clock.
    Distances are capped at max_goal_distance; the inner disk is excluded.
    Left is shelf-local -Y (world -Y for the unrotated shelf); right is +Y.
    Optional goal_angle_range is (low, high) radians of atan2(Y, X); it
    overrides direction/deadzone sampling, retaining distance and edge rules.
    """
    if sweep_direction not in ('both', 'left', 'right'):
        raise ValueError("sweep_direction must be 'both', 'left', or 'right'")
    if not math.isfinite(excluded_x_half_angle) or not 0 <= excluded_x_half_angle < math.pi / 2:
        raise ValueError('Invalid direction deadzone')
    if not math.isfinite(spawn_radius) or spawn_radius < 0 or not math.isfinite(edge_clearance) or edge_clearance < .005:
        raise ValueError('Invalid spawn disk or edge clearance')
    if not math.isfinite(min_goal_distance) or min_goal_distance < 0:
        raise ValueError('Invalid minimum goal distance')
    if not math.isfinite(max_goal_distance) or max_goal_distance < min_goal_distance:
        raise ValueError('Maximum goal distance must be finite and >= minimum')
    if goal_angle_range is not None:
        low, high = goal_angle_range
        if not all(math.isfinite(v) for v in (low, high)) or not 0 < high - low <= 2 * math.pi:
            raise ValueError('Goal angle range must be finite with width in (0, 2*pi]')
    half = bounds[:, :, :2].abs().amax(1)
    safe = bounds.new_tensor([.20, .50]) - half - edge_clearance
    if bool((safe <= spawn_radius).any()):
        raise ValueError('Target footprint leaves no central spawn disk')
    if bool((safe - spawn_radius < min_goal_distance).any()):
        raise ValueError('Minimum goal distance must fit every direction from the spawn disk')
    draws = torch.rand((len(bounds), 4), device=bounds.device, dtype=bounds.dtype, generator=generator)
    angle = 2 * math.pi * draws[:, 0]
    target = spawn_radius * draws[:, 1:2].sqrt() * torch.stack((angle.cos(), angle.sin()), -1)
    if fixed_goal_offset is not None:
        # Keep the parent's spawn distribution and RNG consumption. A fixed
        # shelf-local XY offset supersedes angle/deadzone sampling only.
        if len(fixed_goal_offset) != 2 or not all(math.isfinite(v) for v in fixed_goal_offset):
            raise ValueError('Fixed goal offset must contain two finite XY values')
        distance = math.hypot(*fixed_goal_offset)
        if not min_goal_distance <= distance <= max_goal_distance:
            raise ValueError('Fixed goal offset must lie within the goal distance limits')
        goal = target + bounds.new_tensor(fixed_goal_offset)
        if bool((goal.abs() > safe).any()):
            raise ValueError('Fixed goal offset exceeds the safe shelf bounds')
        return target, goal
    side = (draws[:, 2] >= .5).to(bounds.dtype)
    theta = excluded_x_half_angle + (2 * draws[:, 2] % 1) * (math.pi - 2 * excluded_x_half_angle) + side * math.pi
    if sweep_direction != 'both':
        theta = excluded_x_half_angle + draws[:, 2] * (math.pi - 2 * excluded_x_half_angle)
        if sweep_direction == 'left':
            theta = theta + math.pi
    # Explicit evaluation sectors supersede the training direction deadzone.
    # atan2(Y, X): robot-side +X is 6 o'clock, +Y is 3 o'clock.
    if goal_angle_range is not None:
        theta = low + draws[:, 2] * (high - low)
    direction = torch.stack((theta.cos(), theta.sin()), -1)
    maximum = ((safe - direction.sign() * target) / direction.abs().clamp_min(1e-12)).amin(-1)
    maximum = maximum.clamp_max(max_goal_distance)
    distance = min_goal_distance + (maximum - min_goal_distance) * draws[:, 3]
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
                             target_yaw, obstacle_half, cfg=cfg, requested=requested, generator=generator,
                             cpu_small_batch=True)
    for _ in range(3):
        missing = (~result.complete).nonzero().flatten()
        if not len(missing):
            break
        retry = generate_layout(to_board(target[missing] + offset[missing]), to_board(goal[missing] + offset[missing]),
                                target_half[missing], target_yaw[missing], obstacle_half[missing],
                                cfg=cfg, requested=requested[missing], generator=generator, cpu_small_batch=True)
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


class CircleResetLayout:
    """Cached, scale-aware circles and conservative continuous cell sampling.

    No rejection sampling, count reduction or goal redraw. Geometry is immutable
    for this instance; recreate it after any scale/shape configuration change.
    """
    def __init__(self, bounds, spawn_z, clearance=.01, cell_size=.005, epsilon=1e-6):
        if bounds.ndim != 4 or bounds.shape[1:] != (17, 2, 3):
            raise ValueError('Expected fixed pool bounds [N,17,2,3]')
        if not all(math.isfinite(v) and v > 0 for v in (clearance, cell_size, epsilon)):
            raise ValueError('Circle placement settings must be finite and positive')
        if not bool(torch.isfinite(bounds).all() & (bounds[:, :, 1] > bounds[:, :, 0]).all()):
            raise ValueError('Invalid circle geometry')
        nx, ny = round(.4 / cell_size), round(1. / cell_size)
        if nx < 1 or ny < 1 or abs(nx * cell_size - .4) > 1e-7 or abs(ny * cell_size - 1.) > 1e-7:
            raise ValueError('Cell size must exactly partition the shelf')
        self.bounds, self.spawn_z = bounds, spawn_z
        self.offset = bounds[..., :2].mean(-2)
        self.radius = ((bounds[:, :, 1, :2] - bounds[:, :, 0, :2]) / 2).norm(dim=-1)
        self.expanded = self.radius + clearance
        self.cell_size, self.epsilon = cell_size, epsilon
        self.cell_margin = cell_size / math.sqrt(2.) + epsilon
        x = (torch.arange(nx, device=bounds.device, dtype=bounds.dtype) + .5) * cell_size - .2
        y = (torch.arange(ny, device=bounds.device, dtype=bounds.dtype) + .5) * cell_size - .5
        self.cells = torch.cartesian_prod(x, y)
        self.boundary = torch.minimum(.2 - self.cells[:, 0].abs(), .5 - self.cells[:, 1].abs())
        self._scratch = None

    @staticmethod
    def red_distance(points, start, goal):
        """Distance [B,K] to the extruded segment, including degenerate cases."""
        extension = torch.zeros_like(start)
        extension[:, 0] = (.2 - torch.minimum(start[:, 0], goal[:, 0])).clamp_min(0)
        vertices = torch.stack((start, goal, goal + extension, start + extension), 1)
        p = points[None]
        minimum = torch.full((len(start), len(points)), float('inf'), device=start.device)
        positive = torch.ones_like(minimum, dtype=torch.bool)
        negative = positive.clone()
        for i in range(4):
            a, b = vertices[:, i, None], vertices[:, (i + 1) % 4, None]
            edge, delta = b - a, p - a
            t = (delta * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
            minimum = torch.minimum(minimum, (delta - t.clamp(0, 1)[..., None] * edge).square().sum(-1))
            cross = edge[..., 0] * delta[..., 1] - edge[..., 1] * delta[..., 0]
            positive &= cross >= 0
            negative &= cross <= 0
        area = ((goal[:, 1] - start[:, 1]) * extension[:, 0]).abs()
        inside = (positive | negative) & (area[:, None] > 1e-12)
        return torch.where(inside, 0., minimum.sqrt())

    def sample_commands(self, ids, generator, **command_params):
        """Default goal distribution; specialized layouts may cache another domain."""
        return sample_commands(self.bounds[ids, 0], generator, **command_params)

    @torch.no_grad()
    def sample(self, ids, generator, **command_params):
        bounds = self.bounds[ids]
        n, device = len(ids), bounds.device
        target, goal = self.sample_commands(ids, generator, **command_params)
        requested = torch.randint(0, 7, (n,), device=device, generator=generator)
        selected = torch.rand((n, 16), device=device, generator=generator).argsort(-1)[:, :6] + 1
        radii = self.expanded[ids[:, None], selected]
        # Sort only the selected N IDs; unused candidates must not displace them.
        eligible = torch.arange(6, device=device)[None] < requested[:, None]
        order = torch.where(eligible, radii, -1.).argsort(dim=-1, descending=True, stable=True)
        selected, radii = selected.gather(1, order), radii.gather(1, order)
        draws = torch.rand((n, 6, 4), device=device, generator=generator)
        local = bounds.new_zeros(n, 17, 3)
        local[..., 2] = 50. + 2. * torch.arange(17, device=device)
        local[:, 0, :2], local[:, 0, 2] = target, self.spawn_z[ids, 0]
        yaw = bounds.new_zeros(n, 17)
        yaw[:, 0] = -math.pi / 2
        active = torch.zeros((n, 17), device=device, dtype=torch.bool)
        active[:, 0] = True
        complete = torch.ones(n, device=device, dtype=torch.bool)
        failed_slot = torch.full((n,), -1, device=device, dtype=torch.long)
        # Rotate the target centre offset by the fixed -pi/2 yaw.
        off = self.offset[ids, 0]
        off = torch.stack((off[:, 1], -off[:, 0]), -1)
        working = (requested > 0).nonzero().flatten()
        if len(working) and self._scratch is None:
            # Ordinary tensors also permit explicit reset outside InferenceMode.
            with torch.inference_mode(False):
                self._scratch = (torch.empty((64, len(self.cells)), dtype=torch.bool, device=device),
                                 torch.empty((64, len(self.cells)), dtype=torch.int32, device=device),
                                 torch.empty((6, 64, len(self.cells)), device=device, dtype=bounds.dtype))
        for begin in range(0, len(working), 64):
            rows = working[begin:begin + 64]
            red = self.red_distance(self.cells, target[rows] + off[rows], goal[rows] + off[rows])
            valid_buffer, prefix_buffer, distance_buffer = self._scratch
            valid, prefix = valid_buffer[:len(rows)], prefix_buffer[:len(rows)]
            distances = []
            for slot in range(6):
                radius = radii[rows, slot]
                needed = requested[rows] > slot
                torch.ge(self.boundary[None], radius[:, None] + self.cell_margin, out=valid)
                valid &= red >= self.expanded[ids[rows], 0, None] + radius[:, None] + self.cell_margin
                for previous, distance in enumerate(distances):
                    valid &= distance >= (radius + radii[rows, previous] + self.cell_margin)[:, None].square()
                torch.cumsum(valid, dim=-1, dtype=torch.int32, out=prefix)
                totals = prefix[:, -1]
                ok = (totals > 0) | ~needed
                failed_slot[rows] = torch.where(complete[rows] & ~ok, slot, failed_slot[rows])
                complete[rows] &= ok
                rank = (draws[rows, slot, 0] * totals).long()
                index = torch.searchsorted(prefix, rank.to(torch.int32)[:, None], right=True).squeeze(-1)
                index = index.clamp_max(len(self.cells) - 1)
                centers = self.cells[index] + (draws[rows, slot, 1:3] - .5) * self.cell_size
                angle = draws[rows, slot, 3] * (2 * math.pi) - math.pi
                object_ids = selected[rows, slot]
                offset = self.offset[ids[rows], object_ids]
                rotated = torch.stack((angle.cos() * offset[:, 0] - angle.sin() * offset[:, 1],
                                       angle.sin() * offset[:, 0] + angle.cos() * offset[:, 1]), -1)
                pose = torch.cat((centers - rotated, self.spawn_z[ids[rows], object_ids, None]), -1)
                enabled = needed & complete[rows]
                local[rows, object_ids] = torch.where(enabled[:, None], pose, local[rows, object_ids])
                yaw[rows, object_ids] = torch.where(enabled, angle, 0.)
                active[rows, object_ids] = enabled
                distance = distance_buffer[slot, :len(rows)]
                torch.sum((self.cells[None] - centers[:, None]).square(), dim=-1, out=distance)
                distance.masked_fill_(~enabled[:, None], float('inf'))
                distances.append(distance)
        if not bool(complete.all()):
            bad = ~complete
            raise RuntimeError('Circle reset has no certified free cell: ' + str(dict(
                env_ids=ids[bad].cpu().tolist(), requested=requested[bad].cpu().tolist(),
                selected=selected[bad].cpu().tolist(), radii=radii[bad].cpu().tolist(),
                target=target[bad].cpu().tolist(), goal=goal[bad].cpu().tolist(),
                failed_slot=failed_slot[bad].cpu().tolist(), partial_roots=local[bad].cpu().tolist(),
                seed=generator.initial_seed())))
        return local, yaw, active, goal - target, requested
