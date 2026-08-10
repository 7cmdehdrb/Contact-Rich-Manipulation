"""Cube-relative Reach command extended with a fixed object Sweep goal."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from sweep_rl.shelf_cube_pre_reach.mdp.commands import (
    CubePreReachPoseCommand,
    CubePreReachPoseCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CubeSweepPoseCommand(CubePreReachPoseCommand):
    """Keep the inherited moving Reach point and store one fixed Sweep goal."""

    cfg: "CubeSweepPoseCommandCfg"

    def __init__(self, cfg: "CubeSweepPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg.record_cube_vy_metric:
            self.metrics["cube_vy"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.record_self_collision_metrics:
            self.metrics["self_collision_force_max"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["self_collision_termination_rate"] = torch.zeros(
                self.num_envs, device=self.device
            )

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.record_cube_vy_metric:
            self.metrics["cube_vy"][:] = self.target.data.root_lin_vel_w[:, 1]

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        super()._resample_command(env_ids)
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return
        goal_offset = torch.tensor(
            self.cfg.goal_offset, dtype=torch.float32, device=self.device
        )
        self.goal_pos_w[ids] = self.target.data.root_pos_w[ids] + goal_offset


@configclass
class CubeSweepPoseCommandCfg(CubePreReachPoseCommandCfg):
    """Configuration for :class:`CubeSweepPoseCommand`."""

    class_type: type = CubeSweepPoseCommand
    goal_offset: tuple[float, float, float] = (0.0, 0.18, 0.0)
    record_cube_vy_metric: bool = False
    record_self_collision_metrics: bool = False
