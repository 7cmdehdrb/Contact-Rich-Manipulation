"""Sweep-family contact accounting variants."""
import math

import torch
from isaaclab.utils.math import quat_apply
from .stabilize_sweep_state import SweepOnlyState


class SingleContactSweepState(SweepOnlyState):
    """Two-of-three recognition gives unit contact credit without a ramp."""

    def _init_contact_factor(self, count, device):
        self.contact_recognized = torch.zeros(count, device=device, dtype=torch.bool)

    def _reset_contact_factor(self, ids):
        self.contact_recognized[ids] = False

    def _contact_factor(self, recognized, normalize_contact_count):
        self.contact_recognized.copy_(recognized)
        return self.contact_recognized.float()


class SingleContactV1SweepState(SingleContactSweepState):
    """Child-only moving reach target and signed horizontal-force progress."""

    def __init__(self, count, device):
        super().__init__(count, device)
        self.reach_radius = torch.zeros(count, device=device)
        self.reach_region = torch.zeros(count, device=device, dtype=torch.long)
        self.reach_offset = torch.zeros((count, 3), device=device)
        self.reach_target_center = torch.zeros_like(self.reach_offset)
        self.reach_reference_point = torch.zeros_like(self.reach_offset)
        self.reach_reference_angle = torch.zeros(count, device=device)
        self.reach_distance = torch.zeros(count, device=device)
        self.reach_excess = torch.zeros(count, device=device)
        self.reach_previous_excess = torch.zeros(count, device=device)
        self.reach_delta = torch.zeros(count, device=device)
        self.reach_step = torch.full((count,), -1, device=device, dtype=torch.long)
        self.signed_progress = torch.zeros(count, device=device)
        self.sweep_angle_xy = torch.zeros(count, device=device)
        self.sweep_direction_coefficient = torch.zeros(count, device=device)
        self.sweep_gate = torch.zeros(count, device=device, dtype=torch.bool)
        self.sweep_side_multiplier = torch.ones(count, device=device)
        self.sweep_positive = torch.zeros(count, device=device)
        self.sweep_direction_negative = torch.zeros(count, device=device)

    def reset(self, ids, center, quat, goal):
        super().reset(ids, center, quat, goal)
        self.signed_progress[ids] = 0.
        self.sweep_angle_xy[ids] = 0.
        self.sweep_direction_coefficient[ids] = 0.
        self.sweep_gate[ids] = False
        self.sweep_side_multiplier[ids] = 1.
        self.sweep_positive[ids] = 0.
        self.sweep_direction_negative[ids] = 0.

    def reset_reach(self, ids, center, hand, bounds, direction_shelf, regions, shelf_quat):
        """Called after the final command/placement reset, only for reset rows."""
        radius = ((bounds[:, 1, :2] - bounds[:, 0, :2]) * .5).norm(dim=-1)
        direction = direction_shelf.clone()
        direction[:, 2] = 0.
        norm = direction.norm(dim=-1)
        valid = (torch.isfinite(torch.cat((center, hand, bounds.flatten(1), direction, shelf_quat), -1)).all(-1)
                 & (radius > 0.) & (norm > 1e-12))
        if not bool(valid.all()):
            raise ValueError(f'Invalid v1 reach reset geometry for envs {ids[~valid].tolist()}')
        direction = -direction / norm[:, None]
        angle = math.radians(15.)
        direction[regions == 3] = direction.new_tensor([-math.cos(angle), math.sin(angle), 0.])
        self.reach_radius[ids] = radius
        self.reach_region[ids] = regions.clone()
        self.reach_reference_angle[ids] = torch.atan2(direction[:, 1], -direction[:, 0])
        self.reach_offset[ids] = quat_apply(shelf_quat, radius[:, None] * direction)
        self.reach_target_center[ids] = center
        self.reach_reference_point[ids] = center + self.reach_offset[ids]
        distance = (hand - self.reach_reference_point[ids]).norm(dim=-1)
        self.reach_distance[ids] = distance
        self.reach_excess[ids] = (distance - .05).clamp_min(0.)
        self.reach_previous_excess[ids] = self.reach_excess[ids]
        self.reach_delta[ids] = 0.
        self.reach_step[ids] = -1

    def update_reach(self, center, hand, step):
        """Cache each row's delta once per policy step, including terminal steps."""
        fresh = self.reach_step != step
        point = center[fresh] + self.reach_offset[fresh]
        distance = (hand[fresh] - point).norm(dim=-1)
        if not bool(torch.isfinite(distance).all()):
            ids = fresh.nonzero(as_tuple=False).flatten()
            raise ValueError(f'Nonfinite v1 reach distance for envs {ids[~torch.isfinite(distance)].tolist()}')
        previous = self.reach_excess[fresh].clone()
        excess = (distance - .05).clamp_min(0.)
        self.reach_target_center[fresh] = center[fresh]
        self.reach_reference_point[fresh] = point
        self.reach_distance[fresh] = distance
        self.reach_previous_excess[fresh] = previous
        self.reach_excess[fresh] = excess
        self.reach_delta[fresh] = previous - excess
        self.reach_step[fresh] = step
        return self.reach_delta

    def finish(self, center, goal, mean_force, contact, valid, eef_z, shelf_z, **kwargs):
        distance = (goal[:, :2] - center[:, :2]).norm(dim=-1)
        progress = self.previous_distance - distance
        # Preserve the parent's contact/history, retreat and previous-distance lifecycle.
        super().finish(center, goal, mean_force, contact, valid, eef_z, shelf_z, **kwargs)
        force = mean_force[:, :2]
        magnitude = force.norm(dim=-1)
        usable = (self.contact_recognized & contact & valid & (magnitude >= 1.) & (distance > 1e-12)
                  & torch.isfinite(mean_force).all(-1) & torch.isfinite(distance) & torch.isfinite(progress))
        # Sanitize only masked angle inputs; never interpret invalid force as perfect alignment.
        cosine = (force * (goal[:, :2] - center[:, :2])).sum(-1) / (
            magnitude.clamp_min(1e-12) * distance.clamp_min(1e-12))
        angle = torch.acos(torch.where(usable, cosine, 1.).clamp(-1., 1.))
        coefficient = 2. * torch.exp(-math.log(2.) * (angle / (math.pi / 6)).pow(4)) - 1.
        self.sweep_gate.copy_(usable)
        self.sweep_angle_xy.copy_(torch.where(usable, angle, 0.))
        self.sweep_direction_coefficient.copy_(torch.where(usable, coefficient, 0.))
        self.signed_progress.copy_(torch.where(usable, progress.clamp_min(0.) * coefficient, 0.))
