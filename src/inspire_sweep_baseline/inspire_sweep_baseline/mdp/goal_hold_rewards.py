"""Nonterminal stable-goal achievement, paid at most once per episode."""

import math

import torch

from .goal_terminations import StableGoalSuccess


class StableGoalAchievement(StableGoalSuccess):
    """Track stable occupancy as a reward without terminating the episode.

    RewardManager evaluates this before the hold reward. The inherited timer
    resets on departure, while the achievement latch survives re-entry until
    the environment is reset by timeout or an existing failure.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._awarded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._awarded[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self, env, command_name="target_goal_pos", position_tolerance=0.01,
        speed_tolerance=0.02, upright_angle=math.radians(10.0), hold_time=0.5,
        surface_height=1.05, support_tolerance=0.01, half_extents=(0.20, 0.50),
        object_radius=0.04, failure_terms=(),
    ):
        qualified = super().__call__(
            env, command_name=command_name, position_tolerance=position_tolerance,
            speed_tolerance=speed_tolerance, upright_angle=upright_angle,
            hold_time=hold_time, surface_height=surface_height,
            support_tolerance=support_tolerance, half_extents=half_extents,
            object_radius=object_radius, failure_terms=failure_terms,
        )
        first_achievement = qualified & ~self._awarded
        self._awarded |= qualified
        # RewardManager multiplies by dt: pay the configured weight once.
        return first_achievement.float() / env.step_dt
