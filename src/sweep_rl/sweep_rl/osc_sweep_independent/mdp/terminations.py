"""Safety, contact-continuity, exclusion, and Home success terminations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .common import filtered_contact_mask, target_contact_data_w


def target_invalid_pose(
    env,
    minimum_height: float,
    maximum_tilt: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
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
    robot: Articulation = env.scene[asset_cfg.name]
    wrench = -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]
    return (torch.linalg.norm(wrench[:, :3], dim=-1) > force_limit) | (
        torch.linalg.norm(wrench[:, 3:], dim=-1) > torque_limit
    )


def arm_joint_speed_limit(
    env, maximum_speed: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > maximum_speed, dim=-1
    )


def object_inside_gripper(
    env,
    center_half_extents: tuple[float, float, float],
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Fail when the target center enters the EEF-local exclusion box."""
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    object_pos_eef, _ = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, eef_cfg.body_ids[0]],
        robot.data.body_quat_w[:, eef_cfg.body_ids[0]],
        target.data.root_pos_w,
    )
    extents = torch.tensor(center_half_extents, device=env.device)
    return torch.all(torch.abs(object_pos_eef) <= extents, dim=-1)


def object_disturbed_during_home(
    env,
    command_name: str,
    displacement_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return (command.task_phase == 1) & (
        (displacement > displacement_threshold) | (speed > speed_threshold)
    )


class ContactLostTooLong(ManagerTermBase):
    """Fail after contact has been established and then lost continuously."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ever_contacted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._lost_elapsed = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ever_contacted[env_ids] = False
        self._lost_elapsed[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        sensor_names: tuple[str, ...],
        force_threshold: float,
        maximum_loss_time: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_term(command_name)
        _, _, contact = target_contact_data_w(
            env, sensor_names=sensor_names, force_threshold=force_threshold
        )
        sweeping = command.task_phase == 0
        self._ever_contacted |= contact & sweeping
        actively_lost = sweeping & self._ever_contacted & (~contact)
        self._lost_elapsed[:] = torch.where(
            actively_lost,
            self._lost_elapsed + env.step_dt,
            torch.zeros_like(self._lost_elapsed),
        )
        return actively_lost & (self._lost_elapsed >= maximum_loss_time)


class HomeContactAfterRelease(ManagerTermBase):
    """Allow initial withdrawal, then fail on continued or renewed Home contact."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._home_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._home_elapsed[env_ids] = 0.0
        self._released[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        sensor_name: str,
        force_threshold: float,
        release_grace_time: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_term(command_name)
        home = command.task_phase == 1
        contact = filtered_contact_mask(env, sensor_name, force_threshold)
        self._home_elapsed[:] = torch.where(
            home, self._home_elapsed + env.step_dt, torch.zeros_like(self._home_elapsed)
        )
        self._released |= home & (~contact)
        return home & contact & (
            self._released | (self._home_elapsed >= release_grace_time)
        )


class HomeAfterSweepSuccess(ManagerTermBase):
    """Succeed after a stable OSC return to default arm joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dwell_elapsed = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_elapsed[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        joint_position_threshold: float,
        joint_speed_threshold: float,
        endpoint_threshold: float,
        object_speed_threshold: float,
        object_displacement_threshold: float,
        dwell_time: float,
        contact_sensor_name: str,
        contact_force_threshold: float,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> torch.Tensor:
        command = env.command_manager.get_term(command_name)
        robot: Articulation = env.scene[asset_cfg.name]
        target: RigidObject = env.scene[object_cfg.name]
        joint_error = torch.abs(
            math_utils.wrap_to_pi(
                robot.data.joint_pos[:, asset_cfg.joint_ids]
                - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
            )
        )
        endpoint_error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
        displacement = torch.linalg.norm(
            target.data.root_pos_w - command.parked_object_pos_w, dim=-1
        )
        contact = filtered_contact_mask(
            env, contact_sensor_name, contact_force_threshold
        )
        stable = (
            (command.task_phase == 1)
            & torch.all(joint_error < joint_position_threshold, dim=-1)
            & torch.all(
                torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids])
                < joint_speed_threshold,
                dim=-1,
            )
            & (endpoint_error < endpoint_threshold)
            & (torch.linalg.norm(target.data.root_lin_vel_w, dim=-1) < object_speed_threshold)
            & (displacement < object_displacement_threshold)
            & (~contact)
        )
        self._dwell_elapsed[:] = torch.where(
            stable,
            self._dwell_elapsed + env.step_dt,
            torch.zeros_like(self._dwell_elapsed),
        )
        return self._dwell_elapsed >= dwell_time
