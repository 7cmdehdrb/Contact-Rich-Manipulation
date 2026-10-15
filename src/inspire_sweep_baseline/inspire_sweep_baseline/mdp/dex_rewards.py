"""DexTouch-style historical-best distance improvements for shelf sweeping.

Lee et al. (2024), Sec. IV-B, Eq. (1) and the goal-distance part of Eq. (2)
reward only improvements beyond the closest distance previously achieved.
These terms adapt that rule to the existing EEF and planar sweep goal. Unlike
OmniV2, increments are in meters, without normalization by reset distance.
RewardManager's timestep multiplier is canceled so weights mean reward/meter.
"""

from __future__ import annotations

import math

import torch

from .goal_rewards import PlanarGoalProgress, ReachProgress, _target_state


class DexReachProgress(ReachProgress):
    """New best EEF-to-live-object approach, outside the existing near radius.

    Reuse the reset snapshot and kinematic refresh from ReachProgress. Waiting,
    retreating, and returning to the prior closest distance earn no reward.
    Only the wrist-attached EEF is used, so finger closure alone cannot improve
    this distance. Relative object motion toward the EEF can improve it.
    """

    def __call__(self, env, near_radius=0.10, frame_name="ee_frame"):
        reference = env.scene[frame_name].data.target_pos_w[:, 0]
        distance = (
            torch.linalg.vector_norm(reference - _target_state(env)[:, :3], dim=-1) - near_radius
        ).clamp_min(0.0)
        best = torch.minimum(self._best_distance, distance)
        increment = self._best_distance - best
        self._best_distance[:] = best
        return increment / env.step_dt


class DexPlanarGoalProgress(PlanarGoalProgress):
    """New best object-to-goal XY approach in meters, capped at goal tolerance.

    The inherited reset snapshots object position before commands reset. The
    new command supplies the reset distance on the first call, so the first
    action's improvement is included and old episode goals cannot leak in.
    ``_best_progress`` stores meters gained from reset, not a normalized ratio.
    """

    def __call__(self, env, command_name="target_goal_pos", position_tolerance=0.01):
        if not math.isfinite(position_tolerance) or position_tolerance < 0.0:
            raise ValueError("position_tolerance must be finite and nonnegative")
        goal = env.command_manager.get_command(command_name)[:, :2]
        initial = (
            torch.linalg.vector_norm(self._initial_position - goal, dim=-1) - position_tolerance
        ).clamp_min(0.0)
        distance = (
            torch.linalg.vector_norm(_target_state(env)[:, :2] - goal, dim=-1) - position_tolerance
        ).clamp_min(0.0)
        progress = (initial - distance).clamp_min(0.0)
        best = torch.maximum(self._best_progress, progress)
        increment = best - self._best_progress
        self._best_progress[:] = best
        return increment / env.step_dt
