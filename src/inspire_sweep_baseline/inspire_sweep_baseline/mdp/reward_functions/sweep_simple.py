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


from .dex_rewards import DexPlanarGoalProgress
from .wrench_refactor_relative_ik_test import InverseDistanceReach
from .goal_rewards import _target_state


class AreaMatchedLogGoalProgress(DexPlanarGoalProgress):
    """Log distance potential; match cumulative-curve area for each reset distance.

    Epsilon=1cm, tolerance=0. AUC refers to integral of cumulative reward over
    distance, NOT equal total episode reward (9cm gives 13.92398 vs 9).
    """
    def __call__(self, env, command_name='target_goal_pos', epsilon=.01):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be positive and finite')
        goal = env.command_manager.get_command(command_name)[:, :2]
        initial = (self._initial_position - goal).norm(dim=-1)
        distance = (_target_state(env)[:, :2] - goal).norm(dim=-1)
        # float64 avoids cancellation for very short initial distances. This is
        # tensor-only; the usual reset distribution is 6–12cm.
        d0 = initial.double()
        x = d0 / epsilon
        log = torch.log1p(x)
        denominator = epsilon * (x - log)
        small = x < 1e-3
        series = epsilon * x.square() * (.5 - x / 3 + x.square() / 4)
        denominator = torch.where(small, series, denominator).clamp_min(1e-30)
        coefficient = .5 * d0.square() / denominator
        remaining = torch.minimum(distance.double(), d0)
        progress = (coefficient * (log - torch.log1p(remaining / epsilon))).float().clamp_min(0.)
        best = torch.maximum(self._best_progress, progress)
        increment = best - self._best_progress
        self._best_progress[:] = best
        return increment / env.step_dt


class RetreatReach(InverseDistanceReach):
    def __call__(self, env, near_radius=.10, distance_scale=.10, frame_name='ee_frame', retreat_radius=.02):
        normal = super().__call__(env, near_radius, distance_scale, frame_name)
        state = env._finger_retreat
        active = state.update(env)
        reference = env.scene[frame_name].data.target_pos_w[:, 0]
        gap = ((reference - state.target).norm(dim=-1) - retreat_radius).clamp_min(0.)
        retreat = self._initial_distance / env.cfg.episode_length_s * distance_scale / (distance_scale + gap)
        return torch.where(active, retreat, normal)


class RetreatContactMaintenance(TargetContactDecayMaintenance):
    """No 3cm bypass. Only the latched 2cm/3s gate grants unconditional 20."""
    def __call__(self, env, threshold_n=.01, eef_height_limit=.15):
        reward = super().__call__(env, threshold_n, eef_height_limit)
        active = env._finger_retreat.update(env)
        self._count[active] = 20
        self._misses[active] = 0
        self._cached_reward[active] = 20.
        return torch.where(active, self._cached_reward, reward).clone()
