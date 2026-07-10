"""Joint-action reaching environment."""

from __future__ import annotations

import torch

from src.learning.controllers import JointController
from src.learning.envs.base_reach_env import BaseReachEnv, ReachEnvConfig


class ReachJointEnv(BaseReachEnv):
    """UR5e reaching task where the policy emits six joint-space actions."""

    num_actions = 6

    def __init__(self, cfg: ReachEnvConfig) -> None:
        self.controller = JointController(cfg.action_scale, cfg.max_joint_delta)
        super().__init__(cfg)

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.controller.to_joint_delta(actions), -self.cfg.max_joint_delta, self.cfg.max_joint_delta)
