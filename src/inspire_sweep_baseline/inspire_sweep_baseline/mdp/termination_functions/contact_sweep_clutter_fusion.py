"""Reward and failure reductions over the episode's active object slots.

Zero-disabled goal occupancy and consecutive three-second success."""

from ..common.contact_sweep_clutter import all_object_tilt_angles
from ..common.contact_sweep_clutter_fusion_object_state import active_object_mask
from ..reward_functions.contact_sweep_clutter_fusion import scheduled_goal_hold_xy
from isaaclab.managers import ManagerTermBase
import math as math
import torch as torch

def any_active_object_fall_or_tilt(env, height_condition=1.04, fail_angle=0.9):
    """Apply the existing height/tilt rule to active targets and obstacles."""
    position = env.scene["object_collection"].data.object_pos_w
    height = position[..., 2] - env.scene.env_origins[:, None, 2]
    failed = (height < height_condition) | (all_object_tilt_angles(env) >= fail_angle)
    return (failed & active_object_mask(env, position)).any(dim=-1)


class GoalHoldSuccess(ManagerTermBase):
    """Use the live goal-hold reward predicate, clearing time on exit or reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._hold_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        self._hold_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, hold_time=3.0, failure_terms=()):
        if not math.isfinite(hold_time) or hold_time <= 0:
            raise ValueError("hold_time must be finite and positive")
        goal = env.reward_manager.get_term_cfg("goal_hold")
        eligible = scheduled_goal_hold_xy(env, **goal.params).bool()
        for name in failure_terms:
            eligible &= ~env.termination_manager.get_term(name)
        self._hold_steps[:] = torch.where(eligible, self._hold_steps + 1, 0)
        return self._hold_steps >= math.ceil(hold_time / env.step_dt - 1.0e-9)
