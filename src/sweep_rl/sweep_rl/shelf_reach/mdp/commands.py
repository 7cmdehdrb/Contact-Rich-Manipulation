"""Pose command adapted to the Sweep-Policy Robotiq TCP frame."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TcpUniformPoseCommand(UniformPoseCommand):
    """Uniform pose command whose current-pose metrics use a TCP sensor frame."""

    cfg: "TcpUniformPoseCommandCfg"

    def __init__(self, cfg: "TcpUniformPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.tcp_frame: FrameTransformer = env.scene[cfg.frame_name]

    def _update_metrics(self) -> None:
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        current_pos_w = self.tcp_frame.data.target_pos_w[:, self.cfg.frame_index]
        current_quat_w = self.tcp_frame.data.target_quat_w[:, self.cfg.frame_index]
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            current_pos_w,
            current_quat_w,
        )
        self.metrics["position_error"][:] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"][:] = torch.linalg.norm(rot_error, dim=-1)

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
class TcpUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`TcpUniformPoseCommand`."""

    class_type: type = TcpUniformPoseCommand
    frame_name: str = MISSING
    frame_index: int = 0
