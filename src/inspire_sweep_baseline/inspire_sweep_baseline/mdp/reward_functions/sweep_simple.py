"""Simple-specific contact maintenance with full credit inside the goal."""
import math
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


def finger_action_rate_l2(env, action_group="arm", max_cost=100.0):
    """Parent raw-command squared-sum cost, extended to six hand channels."""
    if not math.isfinite(max_cost) or max_cost < 0.0:
        raise ValueError("max_cost must be finite and nonnegative")
    current, previous = env.action_manager.action, env.action_manager.prev_action
    if current.shape[-1] != 12 or previous.shape != current.shape:
        raise ValueError("Finger penalties require six Arm and six Hand actions")
    if action_group == "arm":
        selection = slice(0, 6)
    elif action_group == "hand":
        selection = slice(6, 12)
    else:
        raise ValueError("action_group must be 'arm' or 'hand'")
    return (current[:, selection] - previous[:, selection]).square().sum(dim=-1).clamp_max(max_cost)
