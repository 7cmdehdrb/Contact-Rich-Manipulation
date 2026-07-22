"""Compact Cartesian force-control rewards for shelf sweeping."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat

from .common import selected_object_state, virtual_ft_wrench_b


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _desired_direction_b(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    direction_b = torch.zeros(env.num_envs, 3, device=env.device)
    direction_b[:, :2] = command[:, :2]
    return direction_b / torch.clamp(
        torch.linalg.norm(direction_b, dim=-1, keepdim=True), min=1.0e-6
    )


def _desired_push_position_w(
    env,
    command_name: str,
    x_offset: float,
    z_offset: float,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    target_pos_w = selected_object_state(env, command_name)[:, :3]
    desired_pos_w = target_pos_w - command.target_width * command.direction_w
    desired_pos_w[:, 0] += x_offset
    desired_pos_w[:, 2] += z_offset
    return desired_pos_w


def _push_position_error(
    env,
    command_name: str,
    x_offset: float,
    z_offset: float,
    ee_frame_cfg,
) -> torch.Tensor:
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    desired_pos_w = _desired_push_position_w(
        env, command_name, x_offset, z_offset
    )
    return torch.linalg.norm(
        desired_pos_w - ee.data.target_pos_w[:, 0], dim=-1
    )


def _proximity_gate(
    env,
    command_name: str,
    near_distance: float,
    far_distance: float,
    x_offset: float,
    z_offset: float,
    ee_frame_cfg,
) -> torch.Tensor:
    if not 0.0 <= near_distance < far_distance:
        raise ValueError("Expected 0 <= near_distance < far_distance.")
    error = _push_position_error(
        env,
        command_name,
        x_offset,
        z_offset,
        ee_frame_cfg,
    )
    return _smoothstep01(
        (far_distance - error) / (far_distance - near_distance)
    )


def _wrist_force_components_b(
    env,
    command_name: str,
    ft_cfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return axial, tangential, and total planar force in base axes."""
    force_b = virtual_ft_wrench_b(env, ft_cfg)[:, :3]
    force_b = force_b.clone()
    force_b[:, 2] = 0.0
    direction_b = _desired_direction_b(env, command_name)
    signed_axial = torch.sum(force_b * direction_b, dim=-1)
    axial = torch.abs(signed_axial)
    tangential_vector = force_b - signed_axial.unsqueeze(-1) * direction_b
    tangential = torch.linalg.norm(tangential_vector, dim=-1)
    total = torch.linalg.norm(force_b, dim=-1)
    return axial, tangential, total


def approach_error(
    env,
    command_name: str,
    shelf_cfg,
    ee_frame_cfg,
    ft_cfg,
    position_scale: float,
    near_distance: float,
    far_distance: float,
    force_low: float,
    force_high: float,
    position_fraction: float,
    x_offset: float,
    z_offset: float,
) -> torch.Tensor:
    """Combine moving push-pose position and orientation errors."""
    if position_scale <= 0.0:
        raise ValueError("position_scale must be positive.")
    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError("position_fraction must lie in [0, 1].")
    if not 0.0 <= force_low < force_high:
        raise ValueError("Expected 0 <= force_low < force_high.")

    shelf: RigidObject = env.scene[shelf_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    position_error = _push_position_error(
        env,
        command_name,
        x_offset,
        z_offset,
        ee_frame_cfg,
    )
    position_value = torch.tanh(position_error / position_scale)

    shelf_rotation = matrix_from_quat(shelf.data.default_root_state[:, 3:7])
    ee_rotation = matrix_from_quat(ee.data.target_quat_w[:, 0])
    alignment = torch.sum(ee_rotation[..., 1] * shelf_rotation[..., 2], dim=-1)
    orientation_value = 0.5 * (1.0 - torch.clamp(alignment, -1.0, 1.0))

    proximity = _proximity_gate(
        env,
        command_name,
        near_distance,
        far_distance,
        x_offset,
        z_offset,
        ee_frame_cfg,
    )
    _, _, total_force = _wrist_force_components_b(env, command_name, ft_cfg)
    force_gate = _smoothstep01(
        (total_force - force_low) / (force_high - force_low)
    )
    contact_gate = proximity * force_gate
    value = (
        position_fraction * position_value
        + (1.0 - position_fraction) * orientation_value
    )

    command = env.command_manager.get_term(command_name)
    command.metrics["push_pose_error_m"][:] = position_error
    command.metrics["contact_gate"][:] = contact_gate
    return (1.0 - contact_gate) * value


class ForceControlError(ManagerTermBase):
    """Combine axial-force tracking and tangential-force suppression."""

    def __call__(
        self,
        env,
        command_name: str,
        ft_cfg,
        ee_frame_cfg,
        near_distance: float,
        far_distance: float,
        tangential_scale: float,
        tangential_fraction: float,
        x_offset: float,
        z_offset: float,
    ) -> torch.Tensor:
        if tangential_scale <= 0.0:
            raise ValueError("tangential_scale must be positive.")
        if not 0.0 <= tangential_fraction <= 1.0:
            raise ValueError("tangential_fraction must lie in [0, 1].")

        axial, tangential, _ = _wrist_force_components_b(
            env, command_name, ft_cfg
        )
        command_value = env.command_manager.get_command(command_name)
        desired_force = command_value[:, 3]
        tolerance = torch.clamp(command_value[:, 4], min=1.0e-3)
        tracking_quality = torch.exp(
            -torch.square((axial - desired_force) / tolerance)
        )
        tangential_error = torch.tanh(tangential / tangential_scale)
        proximity = _proximity_gate(
            env,
            command_name,
            near_distance,
            far_distance,
            x_offset,
            z_offset,
            ee_frame_cfg,
        )
        value = proximity * torch.clamp(
            (1.0 - tracking_quality)
            + tangential_fraction * tangential_error,
            0.0,
            1.0,
        )

        command = env.command_manager.get_term(command_name)
        command.metrics["axial_force_N"][:] = axial
        command.metrics["force_error_N"][:] = torch.abs(axial - desired_force)
        command.metrics["tangential_force_N"][:] = tangential
        return value


class SweepTaskReward(ManagerTermBase):
    """Combine goal-potential progress and low-speed goal stopping."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        progress_fraction: float,
        progress_rate_scale: float,
        position_tolerance: float,
        speed_tolerance: float,
    ) -> torch.Tensor:
        if not 0.0 <= progress_fraction <= 1.0:
            raise ValueError("progress_fraction must lie in [0, 1].")
        if progress_rate_scale <= 0.0:
            raise ValueError("progress_rate_scale must be positive.")
        if position_tolerance <= 0.0 or speed_tolerance <= 0.0:
            raise ValueError("Stopping tolerances must be positive.")

        command = env.command_manager.get_term(command_name)
        state = selected_object_state(env, command_name)
        endpoint_error = torch.linalg.norm(
            state[:, :3] - command.goal_pos_w, dim=-1
        )
        command_distance = torch.clamp(
            env.command_manager.get_command(command_name)[:, 2], min=1.0e-6
        )
        potential = 1.0 - torch.tanh(endpoint_error / command_distance)
        potential_rate = (
            potential - self._previous_potential
        ) / env.step_dt
        progress = torch.tanh(potential_rate / progress_rate_scale)
        progress = torch.where(
            self._initialized, progress, torch.zeros_like(progress)
        )
        self._previous_potential[:] = potential
        self._initialized[:] = True

        speed = torch.linalg.norm(state[:, 7:10], dim=-1)
        stopped = torch.exp(
            -torch.square(endpoint_error / position_tolerance)
        ) * torch.exp(-torch.square(speed / speed_tolerance))
        return progress_fraction * progress + (
            1.0 - progress_fraction
        ) * stopped


class SceneSafetyPenalty(ManagerTermBase):
    """Combine shelf interference and non-target-object motion."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._shelf: RigidObject = env.scene["shelf"]
        self._ee: FrameTransformer = env.scene["ee_frame"]
        self._fingers: FrameTransformer = env.scene["finger_frame"]
        self._wrist: FrameTransformer = env.scene["wrist_frame"]
        self._initial_shelf_pos_w = (
            self._shelf.data.default_root_state[:, :3] + env.scene.env_origins
        )

    def __call__(
        self,
        env,
        command_name: str,
        non_target_speed_scale: float,
    ) -> torch.Tensor:
        if non_target_speed_scale <= 0.0:
            raise ValueError("non_target_speed_scale must be positive.")
        command = env.command_manager.get_term(command_name)
        objects: RigidObjectCollection = env.scene[command.cfg.asset_name]
        speed = torch.linalg.norm(objects.data.object_lin_vel_w, dim=-1)
        env_ids = torch.arange(env.num_envs, device=env.device)
        speed = speed.clone()
        speed[env_ids, command.target_ids] = 0.0
        object_motion = torch.tanh(
            torch.sum(speed, dim=1) / non_target_speed_scale
        )

        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos_w
        shelf_moved = (
            torch.linalg.norm(shelf_delta, dim=-1)
            + torch.linalg.norm(self._shelf.data.root_vel_w, dim=-1)
            > 0.005
        ).float()
        shelf_top = self._shelf.data.root_pos_w.clone()
        shelf_top[:, 2] += 1.06
        near = (
            torch.linalg.norm(
                shelf_top - self._ee.data.target_pos_w[:, 0], dim=-1
            )
            < 0.20
        ).float()
        left_gap = self._fingers.data.target_pos_w[:, 0, 2] - shelf_top[:, 2]
        right_gap = self._fingers.data.target_pos_w[:, 1, 2] - shelf_top[:, 2]
        wrist_gap = self._wrist.data.target_pos_w[:, 0, 2] - shelf_top[:, 2]
        proximity = (
            torch.clamp(1.0 - left_gap / 0.02, 0.0, 1.0)
            + torch.clamp(1.0 - right_gap / 0.02, 0.0, 1.0)
            + torch.clamp(1.0 - wrist_gap / 0.08, 0.0, 1.0)
        ) / 3.0
        shelf_interference = torch.clamp(
            shelf_moved + near * proximity, 0.0, 1.0
        )
        return torch.clamp(object_motion + shelf_interference, 0.0, 1.0)


def action_smoothness(
    env,
    stiffness_fraction: float,
) -> torch.Tensor:
    """Combine normalized Cartesian-pose and stiffness action rates."""
    if not 0.0 <= stiffness_fraction <= 1.0:
        raise ValueError("stiffness_fraction must lie in [0, 1].")
    delta = env.action_manager.action - env.action_manager.prev_action
    delta = torch.clamp(delta, -2.0, 2.0)
    stiffness_rate = 0.25 * torch.mean(torch.square(delta[:, :6]), dim=-1)
    pose_rate = 0.25 * torch.mean(torch.square(delta[:, 6:12]), dim=-1)
    return torch.clamp(
        (1.0 - stiffness_fraction) * pose_rate
        + stiffness_fraction * stiffness_rate,
        0.0,
        1.0,
    )
