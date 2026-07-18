"""Termination conditions for safe sweep training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .rewards import lateral_displacement


def target_reached(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate successfully after reaching the requested endpoint."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    return (endpoint_error < endpoint_threshold) & (
        lateral_displacement(env, command_name) < lateral_threshold
    )


def target_invalid_pose(
    env,
    minimum_height: float,
    maximum_tilt: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the cube falls off the table or tips excessively."""
    target: RigidObject = env.scene[object_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(target.data.root_quat_w)
    return (
        (target.data.root_pos_w[:, 2] < minimum_height)
        | (torch.abs(roll) > maximum_tilt)
        | (torch.abs(pitch) > maximum_tilt)
    )


def excessive_ft_wrench(
    env,
    force_limit: float,
    torque_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate unsafe wrist loads."""
    robot: Articulation = env.scene[asset_cfg.name]
    wrench = -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]
    return (torch.linalg.norm(wrench[:, :3], dim=-1) > force_limit) | (
        torch.linalg.norm(wrench[:, 3:], dim=-1) > torque_limit
    )


def arm_joint_speed_limit(
    env,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate numerically unsafe arm motion."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > maximum_speed,
        dim=-1,
    )


class TargetStoppedAtGoal(ManagerTermBase):
    """Terminate after the object remains stopped at the goal for a dwell time."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dwell_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_elapsed[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        endpoint_threshold: float,
        lateral_threshold: float,
        speed_threshold: float,
        dwell_time: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> torch.Tensor:
        if dwell_time <= 0.0:
            raise ValueError("dwell_time must be positive.")

        command = env.command_manager.get_term(command_name)
        target: RigidObject = env.scene[object_cfg.name]
        endpoint_error = torch.linalg.norm(
            target.data.root_pos_w - command.goal_pos_w, dim=-1
        )
        speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
        stopped_at_goal = (
            (endpoint_error < endpoint_threshold)
            & (lateral_displacement(env, command_name) < lateral_threshold)
            & (speed < speed_threshold)
        )
        self._dwell_elapsed[:] = torch.where(
            stopped_at_goal,
            self._dwell_elapsed + env.step_dt,
            torch.zeros_like(self._dwell_elapsed),
        )
        return self._dwell_elapsed >= dwell_time
