"""Continuous, finite target settling with real terminal-state semantics."""

import math
import torch
from isaaclab.envs.mdp import time_out
from isaaclab.managers import ManagerTermBase


class SettledGoalSuccess(ManagerTermBase):
    """Count once per control step; interrupted conditions clear only that row."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._result = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._last_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._steps[ids] = 0
        self._result[ids] = False

    def __call__(self, env, radius=0.03, speed_limit=0.005, duration=3.0):
        if (not all(math.isfinite(v) for v in (radius, speed_limit, duration, env.step_dt))
                or radius <= 0 or speed_limit < 0 or duration <= 0 or env.step_dt <= 0):
            raise ValueError('Invalid settled-success thresholds')
        step = env.common_step_counter
        if step == self._last_step:
            return self._result
        self._last_step = step
        env.record_goal_accuracy_metrics()
        data = env.scene['object_collection'].data
        rows = torch.arange(env.num_envs, device=env.device)
        target = env.target_id[:, 0]
        position = data.object_pos_w[rows, target]
        velocity = data.object_lin_vel_w[rows, target]
        goal = env.command_manager.get_command('target_goal_pos')
        distance = (position[:, :2] - goal[:, :2]).norm(dim=-1)
        speed = velocity.norm(dim=-1)
        valid = (torch.isfinite(position).all(-1) & torch.isfinite(speed)
                 & torch.isfinite(distance) & (distance <= radius) & (speed <= speed_limit))
        self._steps = torch.where(valid, self._steps + 1, 0)
        required = math.ceil(duration / env.step_dt - 1e-9)
        self._result = self._steps >= required
        # Failure terms precede success in the TerminationManager configuration.
        # A shelf/fall/velocity failure on the final sample must not earn +75.
        self._result &= ~env.termination_manager.terminated
        return self._result


def unsuccessful_timeout(env):
    return time_out(env) & ~env.termination_manager.get_term('target_goal_success')
