"""Cube-relative pre-reach pose command."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.utils import configclass

from sweep_rl.shelf_reach.mdp.commands import (
    TcpUniformPoseCommand,
    TcpUniformPoseCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CubePreReachPoseCommand(TcpUniformPoseCommand):
    """Track a fixed-orientation pose immediately behind the current Cube."""

    cfg: "CubePreReachPoseCommandCfg"

    def __init__(self, cfg: "CubePreReachPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.target: RigidObject = env.scene[cfg.object_name]

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _update_target_position(self, env_ids: torch.Tensor) -> None:
        target_pos_w = self.target.data.root_pos_w[env_ids]
        offset_w = torch.tensor(
            (
                0.0,
                -self.cfg.cube_width * self.cfg.behind_width_scale,
                self.cfg.z_offset,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        desired_pos_w = target_pos_w + offset_w
        desired_pos_b, _ = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            desired_pos_w,
        )
        self.pose_command_b[env_ids, :3] = desired_pos_b

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return
        self._update_target_position(ids)
        count = ids.numel()
        roll = torch.full(
            (count,), self.cfg.target_roll, dtype=torch.float32, device=self.device
        )
        pitch = torch.full(
            (count,), self.cfg.target_pitch, dtype=torch.float32, device=self.device
        )
        yaw = torch.full(
            (count,), self.cfg.target_yaw, dtype=torch.float32, device=self.device
        )
        self.pose_command_b[ids, 3:7] = math_utils.quat_from_euler_xyz(
            roll, pitch, yaw
        )

    def _update_command(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_target_position(env_ids)


@configclass
class CubePreReachPoseCommandCfg(TcpUniformPoseCommandCfg):
    """Configuration for :class:`CubePreReachPoseCommand`."""

    class_type: type = CubePreReachPoseCommand
    object_name: str = MISSING
    cube_width: float = MISSING
    behind_width_scale: float = 1.2
    z_offset: float = 0.03
    target_roll: float = MISSING
    target_pitch: float = 0.0
    target_yaw: float = 0.0
