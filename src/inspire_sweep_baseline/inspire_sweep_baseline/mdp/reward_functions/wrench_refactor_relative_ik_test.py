"""Adapters preserve Holding reward semantics on the independent IK runtime."""
from ..termination_functions.sweep import shelf_collision


def holding_shelf_failure(env):
    """Pay -3 only for Holding's shelf collision predicate, not displacement alone."""
    return shelf_collision(env, threshold=0.1, check_fingers=False, wrist_clearance=0.03).float() / env.step_dt


import math
import torch
from .wrench_refactor import RefactorContactMaintenance
from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w


class TargetContactMaintenance(RefactorContactMaintenance):
    """Target-only 2-of-3 count below the EEF ceiling; no goal occupancy bypass.

    Accepted windows increment the count to 10. Rejected windows pay zero,
    clear the count after reset_after_misses consecutive rejections (default five).
    Inherited reset clears only the selected episodes. RewardManager applies dt.
    """

    def __call__(self, env, threshold_n=0.01, eef_height_limit=0.15, reset_after_misses=5):
        if not math.isfinite(threshold_n) or threshold_n <= 0:
            raise ValueError('Contact threshold must be finite and positive')
        if not math.isfinite(eef_height_limit) or eef_height_limit <= 0:
            raise ValueError('EEF height limit must be finite and positive')
        step = env.common_step_counter
        if step != self._cached_step:
            if self._cached_step is not None and step != self._cached_step + 1:
                self.reset()
            force = target_hand_contact_force_w(env)
            eef_z = env.scene['ee_frame'].data.target_pos_w[:, 0, 2]
            surface_z = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT
            height_valid = (torch.isfinite(eef_z) & torch.isfinite(surface_z)
                            & (eef_z < surface_z + eef_height_limit))
            contact = (height_valid & env.sensor_data_fresh & torch.isfinite(force).all(-1)
                       & (force.norm(dim=-1) >= threshold_n))
            self._history[:, :2] = self._history[:, 1:].clone()
            self._history[:, 2] = contact
            accepted = (self._history.sum(-1) >= 2) & height_valid
            self._misses[:] = torch.where(accepted, 0, self._misses + 1)
            self._count += accepted.long()
            self._count.clamp_(max=10)
            self._count[self._misses >= reset_after_misses] = 0
            self._cached_reward[:] = torch.where(accepted, self._count, 0).float()
            self._cached_step = step
        return self._cached_reward.clone()
