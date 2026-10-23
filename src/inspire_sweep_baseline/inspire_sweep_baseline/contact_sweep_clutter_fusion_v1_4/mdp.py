"""Zero-disabled goal occupancy and consecutive three-second success."""

import math

import torch
from isaaclab.managers import ManagerTermBase

from ..contact_sweep_clutter_v1.mdp import goal_hold_xy


def scheduled_goal_hold_xy(env, position_tolerance=0.0, command_name="target_goal_pos"):
    """A zero radius disables occupancy; positive radii retain v1.0 semantics."""
    if position_tolerance == 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    return goal_hold_xy(env, position_tolerance=position_tolerance, command_name=command_name)


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
