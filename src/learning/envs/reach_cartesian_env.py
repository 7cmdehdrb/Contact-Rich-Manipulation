"""Cartesian-action reaching environment."""

from __future__ import annotations

import torch

from src.learning.controllers import CartesianController
from src.learning.envs.base_reach_env import BaseReachEnv, ReachEnvConfig


class ReachCartesianEnv(BaseReachEnv):
    """UR5e reaching task where the policy emits 3D Cartesian EEF deltas."""

    num_actions = 3

    def __init__(self, cfg: ReachEnvConfig, damping: float = 0.05) -> None:
        self.controller = CartesianController(cfg.action_scale, damping=damping, max_delta=cfg.max_joint_delta)
        super().__init__(cfg)

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        return self.controller.to_joint_delta(actions, self.joint_pos, self.forward_kinematics)
