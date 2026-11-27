"""Independent binary-contact Sweep and signed Reach accounting."""
import torch
import math


def validate_precision_parameters(inner, outer, radius, multiplier):
    """Validate scalar configuration once, outside the tensor step path."""
    if not all(math.isfinite(v) for v in (inner, outer, radius, multiplier)):
        raise ValueError('Sweep precision parameters must be finite')
    if not 0. <= inner < outer or radius <= 0. or multiplier < 1.:
        raise ValueError('Expected 0 <= inner < outer, radius > 0 and multiplier >= 1')


def sweep_force_influence(distance, inner, outer):
    u = ((distance - inner) / (outer - inner)).clamp(0., 1.)
    return u.square() * (3. - 2. * u)


def sweep_distance_sensitivity(distance, radius, multiplier):
    u = (distance / radius).clamp(0., 1.)
    return 1. + (multiplier - 1.) * (1. - u.square() * (3. - 2. * u))


def sweep_weighted_distance(distance, radius, multiplier):
    """Antiderivative of sensitivity, in meters, with H(0) = 0."""
    u = (distance / radius).clamp(0., 1.)
    return distance + (multiplier - 1.) * radius * (u - u.pow(3) + .5 * u.pow(4))


class ProfRevisionState:
    def __init__(self, count, device):
        self.achieved = torch.ones(count, device=device, dtype=torch.bool)
        self.reward_phase = self.achieved
        self.contact_history = torch.zeros((count, 3), device=device, dtype=torch.bool)
        self.contact_recognized = torch.zeros(count, device=device, dtype=torch.bool)
        self.previous_distance = torch.zeros(count, device=device)
        self.positive = torch.zeros_like(self.previous_distance)
        self.retreat = torch.zeros_like(self.previous_distance)
        self.reach_distance = torch.zeros_like(self.previous_distance)
        self.reach_excess = torch.zeros_like(self.previous_distance)
        self.reach_delta = torch.zeros_like(self.previous_distance)
        self.distance = torch.zeros_like(self.previous_distance)
        self.alignment = torch.zeros_like(self.previous_distance)

    def reset(self, ids, center, goal, hand, near_radius):
        self.contact_history[ids] = False
        self.contact_recognized[ids] = False
        self.previous_distance[ids] = (goal[:, :2]-center[:, :2]).norm(dim=-1)
        self.distance[ids] = self.previous_distance[ids]
        self.reach_distance[ids] = (hand-center).norm(dim=-1)
        self.reach_excess[ids] = (self.reach_distance[ids]-near_radius).clamp_min(0.)
        self.positive[ids] = 0.
        self.retreat[ids] = 0.
        self.reach_delta[ids] = 0.
        self.alignment[ids] = 0.

    def finish(self, center, goal, mean_force, contact, valid, hand, near_radius):
        delta = goal[:, :2]-center[:, :2]
        distance = delta.norm(dim=-1)
        progress = self.previous_distance-distance
        self.contact_history[:, 1:] = self.contact_history[:, :-1].clone()
        self.contact_history[:, 0] = contact & valid
        self.contact_recognized.copy_(self.contact_history.sum(-1) >= 2)
        direction = delta / distance[:, None].clamp_min(1e-12)
        alignment = ((mean_force[:, :2]*direction).sum(-1) / mean_force.norm(dim=-1).clamp_min(1e-12)).clamp(0., 1.)
        usable = self._direction_usable(mean_force, contact, valid, distance)
        self.alignment.copy_(torch.where(usable, alignment, 0.))
        self.positive.copy_(progress.clamp_min(0.) * self.contact_recognized * self.alignment)
        self.retreat.copy_((-progress).clamp_min(0.))
        self.previous_distance.copy_(distance)
        self.distance.copy_(distance)
        self.reach_distance.copy_((hand-center).norm(dim=-1))
        excess = (self.reach_distance-near_radius).clamp_min(0.)
        self.reach_delta.copy_(self.reach_excess-excess)
        self.reach_excess.copy_(excess)

    def _direction_usable(self, mean_force, contact, valid, distance):
        return contact & valid & (mean_force[:, :2].norm(dim=-1) >= 1.) & (distance > 1e-12)


class ProfRevisionPrecisionState(ProfRevisionState):
    """Child-only precision caches; all original state fields keep their meaning."""

    def __init__(self, count, device, inner, outer, radius, multiplier):
        validate_precision_parameters(inner, outer, radius, multiplier)
        super().__init__(count, device)
        self.inner, self.outer = inner, outer
        self.radius, self.multiplier = radius, multiplier
        self.force_influence = torch.zeros_like(self.distance)
        self.distance_sensitivity = torch.zeros_like(self.distance)
        self.effective_alignment = torch.zeros_like(self.distance)
        self.weighted_delta = torch.zeros_like(self.distance)
        self.precision_positive = torch.zeros_like(self.distance)
        self.precision_retreat = torch.zeros_like(self.distance)

    def reset(self, ids, center, goal, hand, near_radius):
        super().reset(ids, center, goal, hand, near_radius)
        for buffer in (self.force_influence, self.distance_sensitivity,
                       self.effective_alignment, self.weighted_delta,
                       self.precision_positive, self.precision_retreat):
            buffer[ids] = 0.

    def finish(self, center, goal, mean_force, contact, valid, hand, near_radius):
        previous = self.previous_distance.clone()
        super().finish(center, goal, mean_force, contact, valid, hand, near_radius)
        self.force_influence.copy_(sweep_force_influence(self.distance, self.inner, self.outer))
        self.distance_sensitivity.copy_(sweep_distance_sensitivity(self.distance, self.radius, self.multiplier))
        self.effective_alignment.copy_(1. - self.force_influence + self.force_influence * self.alignment)
        self.weighted_delta.copy_(
            sweep_weighted_distance(previous, self.radius, self.multiplier)
            - sweep_weighted_distance(self.distance, self.radius, self.multiplier))
        gate = self.contact_recognized & contact & valid
        self.precision_positive.copy_(self.weighted_delta.clamp_min(0.) * gate * self.effective_alignment)
        self.precision_retreat.copy_((-self.weighted_delta).clamp_min(0.))


class ProfRevisionContactAlignedState(ProfRevisionPrecisionState):
    """v0_3_4: current valid contact gates direction without a second force floor."""

    def _direction_usable(self, mean_force, contact, valid, distance):
        # ContactSamples already checks normal load >= .01 N. Its summed load
        # cannot be compared to the mean planar resultant: contacts can cancel.
        # A zero resultant has zero alignment via the parent's safe division.
        return contact & valid & (distance > 1e-12)
