"""Bounded, batched OBB placement. Pure torch; no simulator or source-package dependency.

Coordinates: metres, board front at minimum y; wrist projection is world -y.
Inputs/outputs use footprint centres (not necessarily asset root positions).
Only yaw rotations and a fixed-orientation straight target push are modelled.
"""
from dataclasses import dataclass
import math

import torch

RED, GRAY, WHITE = 0, 1, 2


@dataclass(frozen=True)
class LayoutConfig:
    board: tuple = (-.50, .50, 0., .40)
    wall_margin: float = .005
    finger_gap: float = .020
    red_margin: float = .003
    gray_padding: float = .060
    rear_length: float = .160
    rear_width: float = .220
    max_gray: int = 4
    max_white: int = 5
    candidates: int = 32
    restarts: int = 4
    passes: int = 2

    def __post_init__(self):
        if len(self.board) != 4 or not all(math.isfinite(x) for x in self.board):
            raise ValueError('board must be finite (xmin, xmax, ymin, ymax)')
        if not self.board[0] < self.board[1] or not self.board[2] < self.board[3]:
            raise ValueError('board must have positive area')
        for name in ('wall_margin', 'finger_gap', 'red_margin', 'gray_padding',
                     'rear_length', 'rear_width'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('max_gray', 'max_white', 'candidates', 'restarts', 'passes'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name.startswith('max_') else 1):
                raise ValueError(f'Invalid {name}')


@dataclass
class LayoutResult:
    centers: torch.Tensor       # [B, M, 2]; inactive slots are zero, NOT spawn poses
    yaw: torch.Tensor           # [B, M]
    active: torch.Tensor        # [B, M]; caller must park/disable inactive objects
    zones: torch.Tensor         # [M], GRAY slots then WHITE slots
    requested: torch.Tensor     # [B, 2], clipped to configured zone capacities
    actual: torch.Tensor        # [B, 2]
    complete: torch.Tensor      # [B]; false means budget shortfall, NOT infeasibility proof


def axes(yaw):
    """Local x/y unit vectors as rows: [..., 2, 2]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, s), -1), torch.stack((-s, c), -1)), -2)


def footprint_from_bounds(bounds, yaw):
    """Scaled local [..., min/max, xyz] bounds -> (half XY, rotated centre offset).

    Add offset to asset-root XY before planning; subtract it when writing poses.
    Scale must already be applied to bounds, exactly once.
    """
    lo, hi = bounds[..., 0, :2], bounds[..., 1, :2]
    return (hi - lo) / 2, torch.einsum('...i,...ij->...j', (hi + lo) / 2, axes(yaw))


def _radius(half, basis, directions):
    return (torch.abs(torch.einsum('...ai,...ji->...aj', directions, basis)) * half[..., None, :]).sum(-1)


def separated(a, ah, ay, b, bh, by, gap=0.):
    """Broadcastable OBB separation; a separating-axis gap >= gap is required.

    For gap > 0 this is conservative at corners, and guarantees Euclidean gap.
    Boundary touching is allowed only when the requested gap is zero.
    """
    # Four scalar SAT projections avoid a [...,4,2] pairwise matrix allocation.
    ac, ass, bc, bs = ay.cos(), ay.sin(), by.cos(), by.sin()
    c, s = (ac * bc + ass * bs).abs(), (ass * bc - ac * bs).abs()
    delta = a - b
    dx, dy = delta[..., 0], delta[..., 1]
    return (((dx * ac + dy * ass).abs() >= ah[..., 0] + c * bh[..., 0] + s * bh[..., 1] + gap)
            | ((-dx * ass + dy * ac).abs() >= ah[..., 1] + s * bh[..., 0] + c * bh[..., 1] + gap)
            | ((dx * bc + dy * bs).abs() >= bh[..., 0] + c * ah[..., 0] + s * ah[..., 1] + gap)
            | ((-dx * bs + dy * bc).abs() >= bh[..., 1] + s * ah[..., 0] + c * ah[..., 1] + gap))


def _sweep_clearance(centers, half, yaw, target, goal, target_half, target_yaw, cfg):
    # Convex swept OBB + vertical extrusion. Long enough to reach below the board.
    motion = goal - target
    down = torch.zeros_like(motion)
    down[..., 1] = -(cfg.board[3] - cfg.board[2] + 2 * target_half.sum(-1))
    basis = axes(target_yaw)
    generators = torch.cat((basis * target_half[..., :, None], motion[..., None, :] / 2,
                            down[..., None, :] / 2), -2)
    normals = torch.stack((-generators[..., 1], generators[..., 0]), -1)
    norm = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    normals = normals / norm.clamp_min(1e-12)
    normals = normals[:, None].expand(-1, centers.shape[1], -1, -1)
    directions = torch.cat((axes(yaw), normals), -2)
    region_center = (target + goal + down) / 2
    distance = torch.abs(torch.einsum('bki,bkai->bka', centers - region_center[:, None], directions))
    extent = torch.abs(torch.einsum('bgj,bkaj->bkag', generators, directions)).sum(-1)
    extent += _radius(half, axes(yaw), directions)
    valid_axis = torch.linalg.vector_norm(directions, dim=-1) > .5
    return torch.where(valid_axis, distance - extent, -torch.inf).amax(-1)


def zone_masks(centers, half, yaw, target, goal, target_half, target_yaw, cfg):
    """[B,K] candidate masks; whole footprint intersects forbidden/interaction regions.

    RED is swept target + wrist extrusion. GRAY is its padded envelope plus
    the original rear approach rectangle, minus RED. WHITE is the complement.
    Board and initial object-to-object gaps are tested separately.
    """
    clearance = _sweep_clearance(centers, half, yaw, target, goal, target_half, target_yaw, cfg)
    red = clearance < cfg.red_margin
    outer = clearance < cfg.red_margin + cfg.gray_padding
    motion = goal - target
    angle = torch.atan2(motion[:, 1], motion[:, 0])
    direction = axes(angle)[:, 0]
    rear_center = target - direction * cfg.rear_length / 2
    rear_half = torch.empty_like(target)
    rear_half[:, 0], rear_half[:, 1] = cfg.rear_length / 2, cfg.rear_width / 2
    rear = ~separated(centers, half, yaw, rear_center[:, None], rear_half[:, None], angle[:, None])
    if cfg.rear_length == 0 or cfg.rear_width == 0:
        rear = torch.zeros_like(rear)
    gray = ~red & (outer | rear)
    return red, gray, ~red & ~gray


@torch.no_grad()
def generate_layout(target, goal, target_half, target_yaw, obstacle_half, *, cfg=LayoutConfig(),
                    requested=None, generator=None, check_inputs=True):
    """Best-count result of a fixed search budget, vectorized over scenes/restarts/candidates.

    All tensors share floating dtype/device. Shapes: target/goal/target_half [B,2],
    target_yaw [B], obstacle_half [B,max_gray+max_white,2]. Requested [B,2] integer.
    No environment loops, CPU transfers or scalar GPU reads in the search kernel.
    Validation synchronizes; use check_inputs=False only for prevalidated reset inputs.
    Target/goal are never resampled. Failed slots are retried; later/smaller objects
    can still fit. Returns the largest total count; ties favour more Gray objects.
    """
    batch, slots = len(target), cfg.max_gray + cfg.max_white
    dtype, device = target.dtype, target.device
    if target.shape != (batch, 2) or goal.shape != target.shape or target_half.shape != target.shape:
        raise ValueError('Expected target, goal, target_half [B,2]')
    if target_yaw.shape != (batch,) or obstacle_half.shape != (batch, slots, 2):
        raise ValueError('Unexpected yaw or obstacle_half shape')
    if not target.is_floating_point():
        raise ValueError('Floating point geometry required')
    if any(x.device != device or x.dtype != dtype for x in (goal, target_half, target_yaw, obstacle_half)):
        raise ValueError('Geometry tensors must share dtype and device')
    capacity = torch.tensor([cfg.max_gray, cfg.max_white], device=device)
    if requested is None:
        requested = capacity.expand(batch, -1)
    if requested.shape != (batch, 2) or requested.device != device or requested.dtype not in (torch.int32, torch.int64):
        raise ValueError('requested must be integer [B,2] on the geometry device')
    if check_inputs:
        if any(not torch.isfinite(x).all() for x in (target, goal, target_half, target_yaw, obstacle_half)):
            raise ValueError('Nonfinite geometry')
        if (target_half <= 0).any() or (obstacle_half <= 0).any() or (requested < 0).any():
            raise ValueError('Positive half extents and nonnegative counts required')
        extent = (axes(target_yaw).abs() * target_half[..., :, None]).sum(-2)
        low = target.new_tensor([cfg.board[0], cfg.board[2]]) + cfg.wall_margin
        high = target.new_tensor([cfg.board[1], cfg.board[3]]) - cfg.wall_margin
        if any(((p - extent < low) | (p + extent > high)).any() for p in (target, goal)):
            raise ValueError('Target sweep does not fit within board margins')
    requested = requested.clamp_min(0).minimum(capacity)
    zones = torch.cat((torch.full((cfg.max_gray,), GRAY, device=device),
                       torch.full((cfg.max_white,), WHITE, device=device)))
    if not batch or not slots:
        return LayoutResult(target.new_zeros(batch, slots, 2), target.new_zeros(batch, slots),
                            torch.zeros(batch, slots, device=device, dtype=torch.bool), zones,
                            requested, torch.zeros_like(requested), (requested == 0).all(-1))
    repeat = lambda x: x.repeat_interleave(cfg.restarts, dim=0)
    t, g, th, ty, half = map(repeat, (target, goal, target_half, target_yaw, obstacle_half))
    req = repeat(requested)
    n, k = len(t), cfg.candidates
    rows = torch.arange(n, device=device)
    slot_ids = torch.arange(slots, device=device)
    eligible = torch.where(slot_ids < cfg.max_gray, slot_ids < req[:, :1],
                           slot_ids - cfg.max_gray < req[:, 1:])
    positions = target.new_zeros(n, slots, 2)
    rotations = target.new_zeros(n, slots)
    active = torch.zeros(n, slots, dtype=torch.bool, device=device)
    # Different orders per restart prevent a large failed object blocking the rest.
    order = torch.rand(n, slots, device=device, generator=generator).argsort(-1)
    lower = target.new_tensor([cfg.board[0], cfg.board[2]]) + cfg.wall_margin
    upper = target.new_tensor([cfg.board[1], cfg.board[3]]) - cfg.wall_margin
    for _ in range(cfg.passes):
        for step in range(slots):
            slot = order[:, step]
            h = half[rows, slot][:, None]
            yaw = torch.rand(n, k, device=device, dtype=dtype, generator=generator) * (2 * math.pi) - math.pi
            extent = (axes(yaw).abs() * h[..., :, None]).sum(-2)
            lo, hi = lower + extent, upper - extent
            unit = torch.rand(n, k, 2, device=device, dtype=dtype, generator=generator)
            centers = lo + unit * (hi - lo).clamp_min(0)
            red, gray, white = zone_masks(centers, h, yaw, t, g, th, ty, cfg)
            valid = torch.where((slot < cfg.max_gray)[:, None], gray, white) & ~red
            valid &= (hi >= lo).all(-1)
            valid &= separated(centers, h, yaw, t[:, None], th[:, None], ty[:, None], cfg.finger_gap)
            apart = separated(centers[:, :, None], h[:, :, None], yaw[:, :, None],
                              positions[:, None], half[:, None], rotations[:, None], cfg.finger_gap)
            valid &= (apart | ~active[:, None]).all(-1)
            accept = valid.any(-1) & eligible[rows, slot] & ~active[rows, slot]
            first = valid.to(torch.int32).argmax(-1)
            positions[rows, slot] = torch.where(accept[:, None], centers[rows, first], positions[rows, slot])
            rotations[rows, slot] = torch.where(accept, yaw[rows, first], rotations[rows, slot])
            active[rows, slot] |= accept
    counts = torch.stack((active[:, :cfg.max_gray].sum(-1), active[:, cfg.max_gray:].sum(-1)), -1)
    score = counts.sum(-1) * (cfg.max_gray + 1) + counts[:, 0]
    best = score.reshape(batch, cfg.restarts).argmax(-1) + torch.arange(batch, device=device) * cfg.restarts
    actual = counts[best]
    return LayoutResult(positions[best], rotations[best], active[best], zones, requested,
                        actual, (actual == requested).all(-1))
