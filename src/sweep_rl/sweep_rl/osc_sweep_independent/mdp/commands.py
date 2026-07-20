"""Feasible long-distance sweep command with a latched OSC Home phase."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .events import TARGET_SIZE_BUFFER


class FeasibleSweepHomeCommand(CommandTerm):
    """Sample ``[direction_x, direction_y, distance, push_speed]`` safely."""

    cfg: "FeasibleSweepHomeCommandCfg"

    def __init__(self, cfg: "FeasibleSweepHomeCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self.target: RigidObject = env.scene[cfg.object_name]
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self.initial_pose_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.task_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.parked_object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_dwell_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.metrics["endpoint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["home_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["parked_displacement"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _maximum_feasible_distance(
        self,
        position_b: torch.Tensor,
        direction_b: torch.Tensor,
        half_size: torch.Tensor,
    ) -> torch.Tensor:
        margin = self.cfg.workspace_boundary_margin + half_size
        lower_x = self.cfg.workspace_x_range[0] + margin
        upper_x = self.cfg.workspace_x_range[1] - margin
        lower_y = self.cfg.workspace_y_range[0] + margin
        upper_y = self.cfg.workspace_y_range[1] - margin
        infinity = torch.full_like(half_size, float("inf"))
        dx, dy = direction_b[:, 0], direction_b[:, 1]
        tx = torch.where(
            dx > 1.0e-6,
            (upper_x - position_b[:, 0]) / dx,
            torch.where(dx < -1.0e-6, (lower_x - position_b[:, 0]) / dx, infinity),
        )
        ty = torch.where(
            dy > 1.0e-6,
            (upper_y - position_b[:, 1]) / dy,
            torch.where(dy < -1.0e-6, (lower_y - position_b[:, 1]) / dy, infinity),
        )
        return torch.clamp(torch.minimum(tx, ty), min=0.0)

    def _sample_feasible_direction(
        self, position_b: torch.Tensor, half_size: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = position_b.shape[0]
        direction = torch.zeros(count, 2, device=self.device)
        maximum = torch.zeros(count, device=self.device)
        pending = torch.ones(count, dtype=torch.bool, device=self.device)

        for _ in range(self.cfg.maximum_sampling_attempts):
            pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            if pending_ids.numel() == 0:
                break
            angle = torch.empty(pending_ids.numel(), device=self.device).uniform_(
                *self.cfg.direction_angle_range
            )
            candidate = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
            candidate_maximum = self._maximum_feasible_distance(
                position_b[pending_ids], candidate, half_size[pending_ids]
            )
            valid = candidate_maximum >= self.cfg.distance_range[0]
            accepted = pending_ids[valid]
            direction[accepted] = candidate[valid]
            maximum[accepted] = candidate_maximum[valid]
            pending[accepted] = False

        if torch.any(pending):
            pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            center = torch.tensor(
                (
                    0.5 * sum(self.cfg.workspace_x_range),
                    0.5 * sum(self.cfg.workspace_y_range),
                ),
                device=self.device,
            )
            fallback = center.unsqueeze(0) - position_b[pending_ids, :2]
            fallback_norm = torch.linalg.norm(fallback, dim=-1, keepdim=True)
            center_case = fallback_norm.squeeze(-1) < 1.0e-6
            fallback /= torch.clamp(fallback_norm, min=1.0e-6)
            fallback[center_case] = torch.tensor((1.0, 0.0), device=self.device)
            fallback_maximum = self._maximum_feasible_distance(
                position_b[pending_ids], fallback, half_size[pending_ids]
            )
            if torch.any(fallback_maximum < self.cfg.distance_range[0]):
                raise RuntimeError("Target reset pose leaves no feasible minimum-distance sweep.")
            direction[pending_ids] = fallback
            maximum[pending_ids] = fallback_maximum
        return direction, maximum

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w[ids],
            self.robot.data.root_quat_w[ids],
            self.target.data.root_pos_w[ids],
            self.target.data.root_quat_w[ids],
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(object_quat_b)
        self.initial_pose_b[ids] = torch.cat(
            (object_pos_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
        )
        if not hasattr(self._env, TARGET_SIZE_BUFFER):
            raise RuntimeError("Target-size randomization must run before command sampling.")
        half_size = 0.5 * getattr(self._env, TARGET_SIZE_BUFFER)[ids]
        direction_b, maximum = self._sample_feasible_direction(object_pos_b, half_size)
        upper_distance = torch.minimum(
            maximum,
            torch.full_like(maximum, self.cfg.distance_range[1]),
        )
        distance = self.cfg.distance_range[0] + torch.rand_like(upper_distance) * (
            upper_distance - self.cfg.distance_range[0]
        )
        speed = torch.empty(ids.numel(), device=self.device).uniform_(
            *self.cfg.target_speed_range
        )
        self._command[ids] = torch.cat(
            (direction_b, distance.unsqueeze(-1), speed.unsqueeze(-1)), dim=-1
        )

        direction_b_3d = torch.cat(
            (direction_b, torch.zeros(ids.numel(), 1, device=self.device)), dim=-1
        )
        direction_w = math_utils.quat_apply(
            self.robot.data.root_quat_w[ids], direction_b_3d
        )
        self.direction_w[ids] = direction_w
        self.goal_pos_b[ids] = object_pos_b + direction_b_3d * distance.unsqueeze(-1)
        self.goal_pos_w[ids] = (
            self.target.data.root_pos_w[ids] + direction_w * distance.unsqueeze(-1)
        )
        self.task_phase[ids] = 0
        self._goal_dwell_elapsed[ids] = 0.0
        self.parked_object_pos_w[ids] = self.target.data.root_pos_w[ids]

    def _update_metrics(self) -> None:
        endpoint_error = torch.linalg.norm(
            self.target.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        forward_speed = torch.sum(
            self.target.data.root_lin_vel_w * self.direction_w, dim=-1
        )
        current_pos_b, _ = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.target.data.root_pos_w,
        )
        progress = torch.sum(
            (current_pos_b[:, :2] - self.initial_pose_b[:, :2]) * self._command[:, :2],
            dim=-1,
        )
        self.metrics["endpoint_error"][:] = endpoint_error
        self.metrics["speed_error"][:] = torch.abs(forward_speed - self._command[:, 3])
        self.metrics["progress_ratio"][:] = progress / torch.clamp(
            self._command[:, 2], min=1.0e-6
        )
        self.metrics["home_phase"][:] = self.task_phase.float()
        parked_displacement = torch.linalg.norm(
            self.target.data.root_pos_w - self.parked_object_pos_w, dim=-1
        )
        self.metrics["parked_displacement"][:] = torch.where(
            self.task_phase == 1, parked_displacement, torch.zeros_like(parked_displacement)
        )

    def _update_command(self) -> None:
        endpoint_error = torch.linalg.norm(
            self.target.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        object_speed = torch.linalg.norm(self.target.data.root_lin_vel_w, dim=-1)
        parked = (
            (self.task_phase == 0)
            & (endpoint_error < self.cfg.endpoint_threshold)
            & (object_speed < self.cfg.speed_threshold)
        )
        self._goal_dwell_elapsed[:] = torch.where(
            parked,
            self._goal_dwell_elapsed + self._env.step_dt,
            torch.where(
                self.task_phase == 0,
                torch.zeros_like(self._goal_dwell_elapsed),
                self._goal_dwell_elapsed,
            ),
        )
        entering_home = (
            (self.task_phase == 0)
            & (self._goal_dwell_elapsed >= self.cfg.goal_dwell_time)
        )
        self.parked_object_pos_w[entering_home] = self.target.data.root_pos_w[entering_home]
        self.task_phase[:] = torch.where(
            entering_home, torch.ones_like(self.task_phase), self.task_phase
        )


@configclass
class FeasibleSweepHomeCommandCfg(CommandTermCfg):
    class_type: type = FeasibleSweepHomeCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    direction_angle_range: tuple[float, float] = (-math.pi, math.pi)
    distance_range: tuple[float, float] = (0.12, 0.35)
    target_speed_range: tuple[float, float] = (0.04, 0.12)
    workspace_x_range: tuple[float, float] = (0.18, 0.82)
    workspace_y_range: tuple[float, float] = (-0.36, 0.36)
    workspace_boundary_margin: float = 0.015
    maximum_sampling_attempts: int = 32
    endpoint_threshold: float = 0.025
    speed_threshold: float = 0.020
    goal_dwell_time: float = 0.30
