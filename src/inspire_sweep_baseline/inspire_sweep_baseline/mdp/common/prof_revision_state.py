"""Independent binary-contact Sweep and signed Reach accounting."""
import torch


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
        usable = contact & valid & (mean_force[:, :2].norm(dim=-1) >= 1.) & (distance > 1e-12)
        self.alignment.copy_(torch.where(usable, alignment, 0.))
        self.positive.copy_(progress.clamp_min(0.) * self.contact_recognized * self.alignment)
        self.retreat.copy_((-progress).clamp_min(0.))
        self.previous_distance.copy_(distance)
        self.distance.copy_(distance)
        self.reach_distance.copy_((hand-center).norm(dim=-1))
        excess = (self.reach_distance-near_radius).clamp_min(0.)
        self.reach_delta.copy_(self.reach_excess-excess)
        self.reach_excess.copy_(excess)
