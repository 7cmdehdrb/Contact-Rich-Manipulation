"""Cached shelf domain and direct full-direction goals for Simple Hard."""
import math

import torch

from .wrench_refactor_layout import CircleResetLayout


class HardCircleResetLayout(CircleResetLayout):
    """Keep Fast obstacle placement; replace only its target/goal sampler."""

    def configure_commands(self, spawn_radius, edge_clearance, excluded_x_half_angle,
                           min_goal_distance, max_goal_distance):
        """Precompute root-position bounds once, using the target's fixed -pi/2 yaw."""
        values = (spawn_radius, edge_clearance, min_goal_distance, max_goal_distance)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Hard goal settings must be finite')
        if (spawn_radius < 0 or edge_clearance < .005 or min_goal_distance <= 0
                or max_goal_distance < min_goal_distance or excluded_x_half_angle != 0.):
            raise ValueError('Hard goals require valid distances and zero direction deadzone')
        self._command_params = dict(
            spawn_radius=spawn_radius, edge_clearance=edge_clearance,
            excluded_x_half_angle=excluded_x_half_angle,
            min_goal_distance=min_goal_distance, max_goal_distance=max_goal_distance)
        lower, upper = self.bounds[:, 0, 0, :2], self.bounds[:, 0, 1, :2]
        rotated_lower = torch.stack((lower[:, 1], -upper[:, 0]), -1)
        rotated_upper = torch.stack((upper[:, 1], -lower[:, 0]), -1)
        shelf_half = self.bounds.new_tensor((.2, .5))
        self._goal_lower = -shelf_half + edge_clearance - rotated_lower
        self._goal_upper = shelf_half - edge_clearance - rotated_upper
        # This is the only device-side feasibility validation: at startup.
        # A short available distance is valid; it need not reach the requested minimum.
        if not bool(((self._goal_lower < -spawn_radius)
                     & (self._goal_upper > spawn_radius)).all()):
            raise ValueError('Target footprint leaves no central spawn disk for Hard goals')

    def sample_commands(self, ids, generator, **command_params):
        # Scalar configuration check only; no GPU readback or geometry scan on reset.
        if command_params != self._command_params:
            raise RuntimeError('Hard goal settings changed after startup; rebuild the environment')
        params = self._command_params
        draws = torch.rand((len(ids), 4), device=self.bounds.device,
                           dtype=self.bounds.dtype, generator=generator)
        angle = 2 * math.pi * draws[:, 0]
        target = params['spawn_radius'] * draws[:, 1:2].sqrt() * torch.stack(
            (angle.cos(), angle.sin()), -1)
        # Match the parent's zero-deadzone angle mapping and four random draws.
        side = (draws[:, 2] >= .5).to(draws.dtype)
        theta = (2 * draws[:, 2] % 1) * math.pi + side * math.pi
        direction = torch.stack((theta.cos(), theta.sin()), -1)
        to_edge = torch.where(direction >= 0, self._goal_upper[ids] - target,
                              target - self._goal_lower[ids])
        maximum = (to_edge / direction.abs().clamp_min(1e-12)).amin(-1)
        sampled = params['min_goal_distance'] + (
            params['max_goal_distance'] - params['min_goal_distance']) * draws[:, 3]
        # If 15 cm does not fit, use the farthest feasible point, even below 7 cm.
        distance = torch.where(maximum < params['max_goal_distance'], maximum, sampled)
        return target, target + direction * distance[:, None]
