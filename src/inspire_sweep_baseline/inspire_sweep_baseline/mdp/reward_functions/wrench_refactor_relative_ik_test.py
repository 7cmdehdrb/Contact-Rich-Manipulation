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


class TargetContactDecayMaintenance(RefactorContactMaintenance):
    """Target-only 2-of-3 windows: increment to 20, decay by two on rejection.

    Rejected windows pay zero but retain the decayed count for reacquisition.
    Episode reset clears count and history.
    """

    def __call__(self, env, threshold_n=0.01, eef_height_limit=0.15):
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
            self._count[:] = torch.where(accepted, self._count + 1, self._count - 2).clamp(0, 20)
            self._cached_reward[:] = torch.where(accepted, self._count, 0).float()
            self._cached_step = step
        return self._cached_reward.clone()


from .wrench_refactor import RelativeInverseDistance
from ..termination_functions.goal_terminations import _object_position
from ..common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name


class ContactConditionedSweep(RelativeInverseDistance):
    """Policy-step increments with instantaneous target contact, never contact history.

    Each configured term owns the same previous-position/cache lifecycle as inverse
    distance. Slot zero is the fixed target in this Fast environment family.
    """

    def __call__(self, env, component='approach', threshold_n=.01, eef_height_limit=.15,
                 command_name='target_goal_pos'):
        if component not in ('approach', 'direction'):
            raise ValueError('Sweep component must be approach or direction')
        if not math.isfinite(threshold_n) or threshold_n <= 0:
            raise ValueError('Contact threshold must be finite and positive')
        if not math.isfinite(eef_height_limit) or eef_height_limit <= 0:
            raise ValueError('EEF height limit must be finite and positive')
        fresh = self._step != env.common_step_counter
        position = _object_position(env)[:, :2]
        if component == 'approach':
            goal = env.command_manager.get_command(command_name)[:, :2]
            increment = ((self._previous - goal).norm(dim=-1)
                         - (position - goal).norm(dim=-1)).clamp_min(0.)
        else:
            delta = position - self._previous
            direction = env.sweep_dir[:, :2]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            increment = (delta.norm(dim=-1) - (direction * delta).sum(-1)).clamp_min(0.)
        # Read only the target's filtered hand sensor. Sum magnitudes so opposed
        # contact pairs do not cancel. This does not change existing contact terms.
        name = env.scene['object_collection'].object_names[0]
        matrix = env.scene[object_hand_sensor_name(name)].data.force_matrix_w
        if (matrix is None or matrix.ndim != 4 or matrix.shape[0] != env.num_envs
                or matrix.shape[1] != 1 or matrix.shape[2] < 1 or matrix.shape[-1] != 3):
            raise RuntimeError('Target/hand sensor must provide [num_envs,1,filters,3]')
        force = matrix.norm(dim=-1).sum(dim=(1, 2))
        eef_z = env.scene['ee_frame'].data.target_pos_w[:, 0, 2]
        surface_z = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT
        contact = (env.sensor_data_fresh & torch.isfinite(force) & (force >= threshold_n)
                   & torch.isfinite(eef_z) & torch.isfinite(surface_z)
                   & (eef_z < surface_z + eef_height_limit))
        value = torch.where(contact, increment, 0.) / env.step_dt
        self._value[fresh] = value[fresh]
        # Always advance, even without contact, to avoid paying for old free motion.
        self._previous[fresh] = position[fresh]
        self._step[fresh] = env.common_step_counter
        return self._value.clone()
