"""Stable joint actions for the shelf Cube Sweep task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionToLimitsActionCfg,
)
from isaaclab.envs.mdp.actions.joint_actions_to_limits import (
    JointPositionToLimitsAction,
)
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RateLimitedJointPositionToLimitsAction(JointPositionToLimitsAction):
    """Map normalized actions to all joint limits, then slew the target."""

    cfg: "RateLimitedJointPositionToLimitsActionCfg"

    def __init__(
        self,
        cfg: "RateLimitedJointPositionToLimitsActionCfg",
        env: "ManagerBasedEnv",
    ) -> None:
        super().__init__(cfg, env)
        self._max_delta = torch.full(
            (self.num_envs, self.action_dim),
            float("nan"),
            device=self.device,
        )
        indices, _, values = string_utils.resolve_matching_names_values(
            cfg.max_delta,
            self._joint_names,
            preserve_order=cfg.preserve_order,
        )
        self._max_delta[:, indices] = torch.tensor(values, device=self.device)
        if torch.isnan(self._max_delta).any():
            missing = [
                name
                for index, name in enumerate(self._joint_names)
                if torch.isnan(self._max_delta[0, index])
            ]
            raise ValueError(f"No per-step target limit configured for joints: {missing}")
        if torch.any(self._max_delta <= 0.0):
            raise ValueError("All per-step joint target limits must be positive.")
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        if not torch.isfinite(joint_limits).all():
            raise ValueError("Full-range absolute actions require finite joint limits.")
        self._processed_actions[:] = self._asset.data.joint_pos[:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        """Retain the full-range request but bound its change from the last target."""
        previous_target = self._processed_actions.clone()
        super().process_actions(actions)
        requested_target = self._processed_actions.clone()
        target_delta = torch.clamp(
            requested_target - previous_target,
            min=-self._max_delta,
            max=self._max_delta,
        )
        self._processed_actions[:] = previous_target + target_delta

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restart the limiter at the reset pose instead of a stale target."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = self._asset.data.joint_pos[env_ids][
            :, self._joint_ids
        ]


@configclass
class RateLimitedJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    """Configuration for full-range absolute targets with a per-step slew limit."""

    class_type: type = RateLimitedJointPositionToLimitsAction
    max_delta: dict[str, float] = {".*": 0.05}
