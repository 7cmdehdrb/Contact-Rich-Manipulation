"""Wrench OBB geometry and startup-only per-environment layout bank."""

import math
import torch
from .bbox_scene_layout import LayoutConfig, footprint_from_bounds, generate_layout
from .revision_layout import sample_revision_commands

# All joint population/deadzone states reached by the Wrench curricula.
LAYOUT_STAGES = ((30, 3, 3), (25, 3, 4), (25, 4, 4), (20, 4, 4), (20, 4, 5),
                 (15, 4, 5), (10, 4, 5), (5, 4, 5), (0, 4, 5))


class WrenchLayoutBank:
    """Finite per-environment banks, sampled with replacement on every reset.

    Geometry is specific to each environment's fixed shapes/scales. All scheduled
    difficulty states are built at startup, so curriculum transitions never run
    the OBB search in collection. Observation noise is still drawn at every reset.
    """

    @torch.no_grad()
    def __init__(self, bounds, generator, *, size=16, batch_size=256, stages=LAYOUT_STAGES, **params):
        if type(size) is not int or size < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError('Layout bank size and build batch must be positive integers')
        self.size = size
        self.stages = tuple(stages)
        self.banks = {}
        count = len(bounds)
        for degrees, gray, white in self.stages:
            buffers = None
            for start in range(0, count * size, batch_size):
                flat = torch.arange(start, min(start + batch_size, count * size), device=bounds.device)
                output = wrench_layout(bounds[flat // size], generator, gray, white,
                                       excluded_x_half_angle=math.radians(degrees), **params)
                values = (*output[:4], output[4].actual, output[4].complete)
                if buffers is None:
                    buffers = [v.new_empty((count * size, *v.shape[1:])) for v in values]
                for buffer, value in zip(buffers, values):
                    buffer[flat] = value
            if buffers is None:
                raise ValueError('Layout bank requires at least one environment')
            self.banks[(degrees, gray, white)] = tuple(b.reshape(count, size, *b.shape[1:]) for b in buffers)

    def sample(self, ids, generator, gray, white, angle):
        degrees = round(math.degrees(angle), 6)
        key = (degrees, gray, white)
        if key not in self.banks:
            raise ValueError(f'Uninitialized Wrench layout stage: {key}')
        choices = torch.randint(self.size, (len(ids),), device=ids.device, generator=generator)
        return tuple(value[ids, choices] for value in self.banks[key])


def wrench_layout(bounds, generator, gray=3, white=3, *, red_margin=.023, rear_offset=.02, **command_params):
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
    cfg = LayoutConfig(gray_padding=.065, red_margin=red_margin, rear_offset=rear_offset, max_gray=4, max_white=5)
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
