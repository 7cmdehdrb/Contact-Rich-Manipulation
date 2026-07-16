"""Episode-level desired sweep command."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .common import target_contact_data_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SweepMotionCommand(CommandTerm):
    """Sample direction, distance, desired force, and force tolerance at reset.

    The public command is five-dimensional:

    ``[direction_x, direction_y, distance_m, force_N, force_tolerance_N]``.
    """

    cfg: "SweepMotionCommandCfg"

    def __init__(
        self, cfg: "SweepMotionCommandCfg", env: ManagerBasedRLEnv
    ):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.robot_name]
        self._object: RigidObject = env.scene[cfg.object_name]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self.initial_pose_b = torch.zeros(
            self.num_envs, 6, device=self.device
        )
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["endpoint_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["force_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        if count == 0:
            return

        angle = torch.empty(count, device=self.device).uniform_(
            *self.cfg.direction_angle_range
        )
        direction_b = torch.stack(
            (torch.cos(angle), torch.sin(angle)), dim=-1
        )
        distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.distance_range
        )
        force = torch.empty(count, device=self.device).uniform_(
            *self.cfg.force_range
        )
        force_tolerance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.force_tolerance_range
        )
        self._command[env_ids] = torch.cat(
            (
                direction_b,
                distance.unsqueeze(-1),
                force.unsqueeze(-1),
                force_tolerance.unsqueeze(-1),
            ),
            dim=-1,
        )

        object_pos_w = self._object.data.root_pos_w[env_ids]
        object_quat_w = self._object.data.root_quat_w[env_ids]
        root_pos_w = self._robot.data.root_pos_w[env_ids]
        root_quat_w = self._robot.data.root_quat_w[env_ids]
        object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            object_pos_w,
            object_quat_w,
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(object_quat_b)
        self.initial_pose_b[env_ids] = torch.cat(
            (
                object_pos_b,
                torch.stack((roll, pitch, yaw), dim=-1),
            ),
            dim=-1,
        )

        direction_b_3d = torch.cat(
            (
                direction_b,
                torch.zeros(count, 1, device=self.device),
            ),
            dim=-1,
        )
        direction_w = math_utils.quat_apply(root_quat_w, direction_b_3d)
        self.direction_w[env_ids] = direction_w
        self.goal_pos_b[env_ids] = (
            object_pos_b + direction_b_3d * distance.unsqueeze(-1)
        )
        self.goal_pos_w[env_ids] = (
            object_pos_w + direction_w * distance.unsqueeze(-1)
        )

    def _update_metrics(self):
        self.metrics["endpoint_error"][:] = torch.linalg.norm(
            self._object.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        _, force_w, _ = target_contact_data_w(
            self._env, self.cfg.contact_sensor_names
        )
        measured_force = torch.linalg.norm(force_w, dim=-1)
        self.metrics["force_error"][:] = torch.abs(
            measured_force - self._command[:, 3]
        )

    def _update_command(self):
        pass


@configclass
class SweepMotionCommandCfg(CommandTermCfg):
    """Configuration for episode-level sweep commands."""

    class_type: type = SweepMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    direction_angle_range: tuple[float, float] = (-math.pi, math.pi)
    distance_range: tuple[float, float] = (0.10, 0.22)
    force_range: tuple[float, float] = (8.0, 25.0)
    force_tolerance_range: tuple[float, float] = (3.0, 6.0)
    contact_sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    )
