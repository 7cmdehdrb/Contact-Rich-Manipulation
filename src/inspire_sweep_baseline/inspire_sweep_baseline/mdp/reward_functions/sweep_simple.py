"""Simple-specific contact maintenance with full credit inside the goal."""
import torch

from .wrench_refactor_relative_ik_test import TargetContactDecayMaintenance
from ..termination_functions.goal_terminations import _object_position


class GoalContactMaintenance(TargetContactDecayMaintenance):
    """XY distance < 3 cm sets count/reward to 20, regardless of contact or height.

    Outside the goal, retain the parent's real contact history and decay rule.
    Goal occupancy does not fabricate tactile samples in the 2-of-3 history.
    """

    def __call__(self, env, threshold_n=.01, eef_height_limit=.15):
        fresh = self._cached_step != env.common_step_counter
        reward = super().__call__(env, threshold_n, eef_height_limit)
        if fresh:
            goal = env.command_manager.get_command('target_goal_pos')[:, :2]
            distance = (_object_position(env)[:, :2] - goal).norm(dim=-1)
            inside = torch.isfinite(distance) & (distance < .03)
            self._count[inside] = 20
            self._misses[inside] = 0
            self._cached_reward[inside] = 20.
            reward = self._cached_reward.clone()
        return reward
