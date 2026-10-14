"""Dynamic target-object goal command for shelf sweeping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ShelfSweepGoalCommand(CommandTerm):
    """Expose a 5-D Cartesian desired-motion command for the selected object."""

    cfg: "ShelfSweepGoalCommandCfg"

    def __init__(self, cfg: "ShelfSweepGoalCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self.object_collection: RigidObjectCollection = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self.target_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.target_width = torch.full(
            (self.num_envs, 1), cfg.object_widths[0], device=self.device
        )
        self.sweep_direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.sweep_direction_w[:, 1] = cfg.sweep_distance
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_pose_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.initial_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.initial_target_quat_w[:, 0] = 1.0
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["endpoint_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_speed_mps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["push_pose_error_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["contact_gate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["axial_force_N"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_error_N"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tangential_force_N"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        selected = self.object_collection.data.object_state_w[
            env_ids, self.target_ids[env_ids]
        ]
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        position_b, quaternion_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            selected[:, :3],
            selected[:, 3:7],
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion_b)
        self.initial_pose_b[env_ids] = torch.cat(
            (position_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
        )

        direction_w = self.sweep_direction_w[env_ids] / torch.clamp(
            torch.linalg.norm(
                self.sweep_direction_w[env_ids], dim=-1, keepdim=True
            ),
            min=1.0e-6,
        )
        direction_b = math_utils.quat_apply_inverse(root_quat_w, direction_w)
        count = len(env_ids)
        force = torch.empty(count, device=self.device).uniform_(*self.cfg.force_range)
        tolerance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.force_tolerance_range
        )
        self._command[env_ids] = torch.cat(
            (
                direction_b[:, :2],
                torch.full(
                    (count, 1), self.cfg.sweep_distance, device=self.device
                ),
                force.unsqueeze(-1),
                tolerance.unsqueeze(-1),
            ),
            dim=-1,
        )
        self.initial_target_pos_w[env_ids] = selected[:, :3]
        self.initial_target_quat_w[env_ids] = selected[:, 3:7]
        self.direction_w[env_ids] = direction_w
        self.goal_pos_w[env_ids] = selected[:, :3] + self.sweep_direction_w[env_ids]

    def _update_metrics(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        state = self.object_collection.data.object_state_w[env_ids, self.target_ids]
        self.metrics["endpoint_error_m"][:] = torch.linalg.norm(
            state[:, :3] - self.goal_pos_w, dim=-1
        )
        displacement = state[:, :3] - self.initial_target_pos_w
        progress = torch.sum(displacement * self.direction_w, dim=-1)
        self.metrics["progress_ratio"][:] = progress / self.cfg.sweep_distance
        self.metrics["object_speed_mps"][:] = torch.linalg.norm(state[:, 7:10], dim=-1)

    def _update_command(self) -> None:
        pass


@configclass
class ShelfSweepGoalCommandCfg(CommandTermCfg):
    class_type: type = ShelfSweepGoalCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = MISSING
    robot_name: str = MISSING
    object_widths: tuple[float, ...] = MISSING
    sweep_distance: float = 0.18
    force_range: tuple[float, float] = (8.0, 25.0)
    force_tolerance_range: tuple[float, float] = (3.0, 6.0)
