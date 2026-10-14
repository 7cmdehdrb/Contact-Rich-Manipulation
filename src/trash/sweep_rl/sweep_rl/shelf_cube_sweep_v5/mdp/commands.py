"""Cube-relative Reach command and fixed Sweep goal for standalone v5."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CubeSweepPoseCommand(UniformPoseCommand):
    """Track the random-sweep contact point and retain a fixed Cube goal."""

    cfg: "CubeSweepPoseCommandCfg"

    def __init__(self, cfg: "CubeSweepPoseCommandCfg", env: "ManagerBasedEnv"):
        if cfg.cube_width <= 0.0:
            raise ValueError("Cube width must be positive.")
        if cfg.behind_width_scale <= 0.0:
            raise ValueError("Contact-point width scale must be positive.")
        if cfg.push_gate_distance <= 0.0:
            raise ValueError("Push gate distance must be positive.")
        super().__init__(cfg, env)
        self.tcp_frame: FrameTransformer = env.scene[cfg.frame_name]
        self.wrist_frame: FrameTransformer = env.scene[cfg.wrist_frame_name]
        self.target: RigidObject = env.scene[cfg.object_name]
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_target_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reach_target_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._home_joint_ids, _ = self.robot.find_joints(
            cfg.home_joint_names, preserve_order=True
        )
        for metric_name in (
            "goal_xyz_distance",
            "goal_yz_distance",
            "cube_y_displacement",
            "cube_y_delta",
            "cube_y_velocity",
            "normalized_progress_rate",
            "tcp_contact_distance",
            "wrist_y_error",
            "zeta_active",
            "home_gate",
            "home_joint_error_5",
        ):
            self.metrics[metric_name] = torch.zeros(
                self.num_envs, device=self.device
            )

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _contact_position_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.target.data.root_pos_w[env_ids].clone()
        position[:, 0] += self.cfg.x_offset
        position[:, 1] -= self.cfg.cube_width * self.cfg.behind_width_scale
        position[:, 2] += self.cfg.z_offset
        return position

    def _update_target_position(self, env_ids: torch.Tensor) -> None:
        desired_pos_w = self._contact_position_w(env_ids)
        self.reach_target_pos_w[env_ids] = desired_pos_w
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
        self.initial_target_pos_w[ids] = self.target.data.root_pos_w[ids]
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
        goal_offset = torch.tensor(
            self.cfg.goal_offset, dtype=torch.float32, device=self.device
        )
        self.goal_pos_w[ids] = self.initial_target_pos_w[ids] + goal_offset

    def _update_command(self) -> None:
        self._update_target_position(
            torch.arange(self.num_envs, device=self.device)
        )

    def _update_metrics(self) -> None:
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = (
            combine_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.pose_command_b[:, :3],
                self.pose_command_b[:, 3:],
            )
        )
        current_pos_w = self.tcp_frame.data.target_pos_w[:, self.cfg.frame_index]
        current_quat_w = self.tcp_frame.data.target_quat_w[
            :, self.cfg.frame_index
        ]
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            current_pos_w,
            current_quat_w,
        )
        self.metrics["position_error"][:] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"][:] = torch.linalg.norm(
            rot_error, dim=-1
        )

        target_pos_w = self.target.data.root_pos_w
        goal_xyz_distance = torch.linalg.vector_norm(
            self.goal_pos_w - target_pos_w, dim=-1
        )
        goal_yz_distance = torch.linalg.vector_norm(
            self.goal_pos_w[:, 1:] - target_pos_w[:, 1:], dim=-1
        )
        tcp_contact_distance = torch.linalg.vector_norm(
            current_pos_w - self.reach_target_pos_w, dim=-1
        )
        wrist_pos_w = self.wrist_frame.data.target_pos_w[
            :, self.cfg.wrist_frame_index
        ]
        wrist_y_error = torch.abs(
            self.reach_target_pos_w[:, 1] - wrist_pos_w[:, 1]
        )
        zeta_active = (
            (tcp_contact_distance < self.cfg.push_gate_distance)
            & (wrist_y_error < self.cfg.push_gate_distance)
        )
        home_joint_error = torch.sum(
            torch.abs(
                self.robot.data.joint_pos[:, self._home_joint_ids]
                - self.robot.data.default_joint_pos[:, self._home_joint_ids]
            ),
            dim=-1,
        )
        home_gate = 0.5 * (
            1.0
            - torch.tanh(
                self.cfg.home_gate_sharpness
                * (goal_yz_distance - self.cfg.home_success_distance)
            )
        )

        self.metrics["goal_xyz_distance"][:] = goal_xyz_distance
        self.metrics["goal_yz_distance"][:] = goal_yz_distance
        self.metrics["cube_y_displacement"][:] = (
            target_pos_w[:, 1] - self.initial_target_pos_w[:, 1]
        )
        self.metrics["cube_y_velocity"][:] = self.target.data.root_lin_vel_w[:, 1]
        self.metrics["tcp_contact_distance"][:] = tcp_contact_distance
        self.metrics["wrist_y_error"][:] = wrist_y_error
        self.metrics["zeta_active"][:] = zeta_active.float()
        self.metrics["home_gate"][:] = home_gate
        self.metrics["home_joint_error_5"][:] = home_joint_error

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self.robot.is_initialized or not self.tcp_frame.is_initialized:
            return
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:]
        )
        self.current_pose_visualizer.visualize(
            self.tcp_frame.data.target_pos_w[:, self.cfg.frame_index],
            self.tcp_frame.data.target_quat_w[:, self.cfg.frame_index],
        )


@configclass
class CubeSweepPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = CubeSweepPoseCommand
    frame_name: str = MISSING
    frame_index: int = 0
    object_name: str = MISSING
    cube_width: float = MISSING
    behind_width_scale: float = 1.0
    x_offset: float = -0.02
    z_offset: float = 0.09
    push_gate_distance: float = 0.04
    wrist_frame_name: str = MISSING
    wrist_frame_index: int = 0
    target_roll: float = MISSING
    target_pitch: float = 0.0
    target_yaw: float = 0.0
    goal_offset: tuple[float, float, float] = (0.0, 0.18, 0.0)
    home_joint_names: list[str] = MISSING
    home_success_distance: float = 0.03
    home_gate_sharpness: float = 100.0
