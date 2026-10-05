"""The two reaching rewards: distance and executed action changes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase

from ..geometry import control_point_pose_w


def eef_target_distance_reward(env) -> torch.Tensor:
    position_w, _ = control_point_pose_w(env)
    target = env.command_manager.get_term("target_position").target_pos_w
    distance = torch.linalg.vector_norm(position_w - target, dim=-1)
    return torch.exp(-distance / env.cfg.task.goal_reward_sigma_m)


class ExecutedActionRate(ManagerTermBase):
    """Penalize changes in the normalized arm and rate-limited hand actions."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous = torch.zeros((self.num_envs, 8), device=self.device)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_counter = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._cached = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._previous[index] = 0.0
        self._valid[index] = False
        self._last_counter[index] = -1
        self._cached[index] = 0.0

    def __call__(self, env) -> torch.Tensor:
        arm = env.action_manager.get_term("arm_action").raw_actions
        hand = env.action_manager.get_term("hand_action").processed_actions
        executed = torch.cat((arm, hand), dim=-1)
        counter = getattr(env, "_sim_step_counter", getattr(env, "common_step_counter", 0))
        changed = self._last_counter != counter
        difference = executed - self._previous
        value = -(difference[:, :6].square().sum(dim=-1) / 6.0 + difference[:, 6:].square().sum(dim=-1) / 2.0)
        value = torch.where(self._valid, value, torch.zeros_like(value))
        self._cached[changed] = value[changed]
        self._previous[changed] = executed[changed].detach()
        self._valid[changed] = True
        self._last_counter[changed] = counter
        return self._cached.clone()


__all__ = ["ExecutedActionRate", "eef_target_distance_reward"]
