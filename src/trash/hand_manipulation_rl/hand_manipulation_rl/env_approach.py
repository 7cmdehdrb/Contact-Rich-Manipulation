"""Inherited blind-sweep environment with approach-specific geometry helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .env import BlindSweepEnv

if TYPE_CHECKING:
    from .env_approach_cfg import BlindSweepApproachEnvCfg


class BlindSweepApproachEnv(BlindSweepEnv):
    """BlindSweepEnv variant with mode-symmetric hand approach geometry."""

    cfg: "BlindSweepApproachEnvCfg"

    def hand_x_points_up(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return whether Hand +X is nominally aligned with world-up."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        mode_sign = torch.where(
            self.surface_mode[env_ids] == 0,
            torch.ones(len(env_ids), device=self.device),
            -torch.ones(len(env_ids), device=self.device),
        )
        return mode_sign * self.command_direction_w[env_ids, 1] > 0.0

    def safe_approach_height_offset_m(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the orientation-conditioned C height above object center."""

        hand_x_up = self.hand_x_points_up(env_ids)
        return torch.where(
            hand_x_up,
            torch.full_like(
                hand_x_up,
                self.cfg.task.stable_reset_hand_x_up_height_m,
                dtype=torch.float32,
            ),
            torch.full_like(
                hand_x_up,
                self.cfg.task.stable_reset_hand_x_down_height_m,
                dtype=torch.float32,
            ),
        )

    def selected_surface_control_point_w(self) -> torch.Tensor:
        """Return a palm/dorsal-symmetric proxy for the selected hand face.

        The parent's virtual C uses the fixed H-to-C +Y offset, which moves
        away from the object in dorsal mode.  Mirroring only that local Y
        component makes physically equivalent palm and dorsal approaches share
        one reward geometry while retaining C's along-finger offset.
        """

        robot = self.scene["robot"]
        hand_body_id = self._resolve_hand_body()
        hand_position_w = robot.data.body_pos_w[:, hand_body_id]
        hand_quaternion_w = robot.data.body_quat_w[:, hand_body_id]
        local_offset_h = torch.as_tensor(
            self.cfg.task.c_offset_h,
            dtype=hand_position_w.dtype,
            device=self.device,
        ).expand(self.num_envs, -1).clone()
        local_offset_h[:, 1] *= torch.where(
            self.surface_mode == 0,
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        )
        return hand_position_w + math_utils.quat_apply(hand_quaternion_w, local_offset_h)


__all__ = ["BlindSweepApproachEnv"]
