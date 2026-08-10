"""Reward terms owned by the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer


def _desired_pose_w(
    env, command_name: str, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    return math_utils.combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )


def tcp_position_command_reward_exp(
    env,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    desired_pos_w, _ = _desired_pose_w(env, command_name, robot_cfg)
    frame: FrameTransformer = env.scene[frame_cfg.name]
    current_pos_w = frame.data.target_pos_w[:, frame_index]
    distance = torch.linalg.norm(current_pos_w - desired_pos_w, dim=-1)
    return torch.exp(-10.0 * distance)


def signed_axis_alignment(
    ee_y_axis: torch.Tensor, shelf_z_axis: torch.Tensor
) -> torch.Tensor:
    """Return the signed squared alignment used by random sweep."""
    alignment = torch.sum(ee_y_axis * shelf_z_axis, dim=-1)
    return torch.sign(alignment) * alignment.square()


def align_ee_to_shelf(
    env,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    frame_index: int = 0,
) -> torch.Tensor:
    """Align EE local-Y with shelf local-Z exactly as random sweep does."""
    shelf: RigidObject = env.scene[shelf_cfg.name]
    frame: FrameTransformer = env.scene[frame_cfg.name]
    shelf_rotation = math_utils.matrix_from_quat(shelf.data.default_root_state[:, 3:7])
    ee_rotation = math_utils.matrix_from_quat(
        frame.data.target_quat_w[:, frame_index]
    )
    return signed_axis_alignment(ee_rotation[..., 1, :], shelf_rotation[..., 2, :])


def shelf_floor_contact_mask(
    force_matrix_w: torch.Tensor,
    contact_pos_w: torch.Tensor,
    shelf_pos_w: torch.Tensor,
    shelf_quat_w: torch.Tensor,
    *,
    force_threshold: float,
    surface_heights: tuple[float, ...],
    surface_tolerance: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> torch.Tensor:
    if force_threshold <= 0.0 or surface_tolerance <= 0.0:
        raise ValueError("Contact force threshold and tolerance must be positive.")
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("Shelf floor bounds must be strictly increasing.")
    if not surface_heights:
        raise ValueError("At least one shelf surface height is required.")
    if force_matrix_w.shape != contact_pos_w.shape or force_matrix_w.ndim != 4:
        raise ValueError("Contact tensors must have matching (N, B, M, 3) shapes.")

    valid = torch.isfinite(force_matrix_w).all(dim=-1) & torch.isfinite(
        contact_pos_w
    ).all(dim=-1)
    safe_force_w = torch.nan_to_num(force_matrix_w, nan=0.0)
    safe_contact_pos_w = torch.nan_to_num(contact_pos_w, nan=0.0)
    num_bodies, num_filters = force_matrix_w.shape[1:3]
    shelf_pos_w = shelf_pos_w[:, None, None, :]
    shelf_quat_w = shelf_quat_w[:, None, None, :].expand(
        -1, num_bodies, num_filters, -1
    )
    contact_pos_s = math_utils.quat_apply_inverse(
        shelf_quat_w.reshape(-1, 4),
        (safe_contact_pos_w - shelf_pos_w).reshape(-1, 3),
    ).reshape_as(safe_contact_pos_w)
    heights = torch.as_tensor(
        surface_heights, dtype=contact_pos_s.dtype, device=contact_pos_s.device
    )
    on_height = torch.any(
        torch.abs(contact_pos_s[..., 2].unsqueeze(-1) - heights)
        <= surface_tolerance,
        dim=-1,
    )
    on_floor = (
        (contact_pos_s[..., 0] >= x_bounds[0])
        & (contact_pos_s[..., 0] <= x_bounds[1])
        & (contact_pos_s[..., 1] >= y_bounds[0])
        & (contact_pos_s[..., 1] <= y_bounds[1])
        & on_height
    )
    force = torch.linalg.norm(safe_force_w, dim=-1)
    return torch.any(valid & on_floor & (force > force_threshold), dim=(1, 2))


def shelf_collision(
    env,
    sensor_name: str,
    force_threshold: float,
    surface_heights: tuple[float, ...],
    surface_tolerance: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_matrix_w is None or sensor.data.contact_pos_w is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' has no filtered data.")
    return shelf_floor_contact_mask(
        sensor.data.force_matrix_w,
        sensor.data.contact_pos_w,
        shelf.data.root_pos_w,
        shelf.data.root_quat_w,
        force_threshold=force_threshold,
        surface_heights=surface_heights,
        surface_tolerance=surface_tolerance,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    ).float()


def cube_push_offset_position(
    target_position_w: torch.Tensor,
    cube_width: float,
    x_offset: float,
    behind_width_scale: float,
    z_offset: float,
) -> torch.Tensor:
    target_offset_w = target_position_w.clone()
    target_offset_w[:, 0] += x_offset
    target_offset_w[:, 1] -= cube_width * behind_width_scale
    target_offset_w[:, 2] += z_offset
    return target_offset_w


def pushing_target_raw_reward(
    distance: torch.Tensor,
    progress_delta: torch.Tensor,
    contact_distance: torch.Tensor,
    wrist_y_error: torch.Tensor,
    target_velocity: torch.Tensor,
    gate_distance: float,
    velocity_reward_min: float = 0.05,
    velocity_penalty_min: float = 0.10,
) -> torch.Tensor:
    if gate_distance <= 0.0:
        raise ValueError("Sweep gate distance must be positive.")
    if not 0.0 <= velocity_reward_min < velocity_penalty_min:
        raise ValueError("Push velocity thresholds are invalid.")
    zeta_m = (contact_distance < gate_distance) & (wrist_y_error < gate_distance)
    target_speed = torch.abs(target_velocity)
    velocity_reward = torch.where(
        target_speed > velocity_reward_min,
        torch.where(target_speed < velocity_penalty_min, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * torch.exp(-5.0 * distance),
        zeta_m.float() * (progress_delta + velocity_reward),
    )


def cube_upright_quality(
    root_quat_w: torch.Tensor, max_tilt_radians: float
) -> torch.Tensor:
    """Retained utility; v5 intentionally does not call this term."""
    if not 0.0 < max_tilt_radians < 0.5 * torch.pi:
        raise ValueError("Maximum Cube tilt must be between zero and pi/2.")
    local_up = torch.zeros(
        (root_quat_w.shape[0], 3),
        dtype=root_quat_w.dtype,
        device=root_quat_w.device,
    )
    local_up[:, 2] = 1.0
    up_z = math_utils.quat_apply(root_quat_w, local_up)[:, 2]
    min_up_z = torch.cos(
        torch.as_tensor(
            max_tilt_radians, dtype=root_quat_w.dtype, device=root_quat_w.device
        )
    )
    return torch.clamp((up_z - min_up_z) / (1.0 - min_up_z), 0.0, 1.0)


class PushingTargetDeltaReward(ManagerTermBase):
    """Reward incremental signed Sweep progress instead of held position."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        object_cfg = cfg.params.get(
            "object_cfg", SceneEntityCfg("target_object")
        )
        target: RigidObject = env.scene[object_cfg.name]
        self._previous_target_y = target.data.root_pos_w[:, 1].clone()
        self.last_delta_y = torch.zeros(env.num_envs, device=env.device)
        self.last_progress_delta = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        object_cfg = self.cfg.params.get(
            "object_cfg", SceneEntityCfg("target_object")
        )
        target: RigidObject = self._env.scene[object_cfg.name]
        self._previous_target_y[env_ids] = target.data.root_pos_w[env_ids, 1]
        self.last_delta_y[env_ids] = 0.0
        self.last_progress_delta[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        gate_distance: float,
        goal_distance: float,
        velocity_reward_min: float = 0.05,
        velocity_penalty_min: float = 0.10,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
    ) -> torch.Tensor:
        if goal_distance <= 0.0:
            raise ValueError("Sweep goal distance must be positive.")
        command = env.command_manager.get_term(command_name)
        target: RigidObject = env.scene[object_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        wrist_frame: FrameTransformer = env.scene[wrist_frame_cfg.name]

        current_target_y = target.data.root_pos_w[:, 1]
        delta_y = current_target_y - self._previous_target_y
        self._previous_target_y.copy_(current_target_y)
        sweep_direction = torch.sign(
            command.goal_pos_w[:, 1] - command.initial_target_pos_w[:, 1]
        )
        progress_delta = sweep_direction * delta_y / (goal_distance * env.step_dt)
        self.last_delta_y[:] = delta_y
        self.last_progress_delta[:] = progress_delta
        command.metrics["cube_y_delta"][:] = delta_y
        command.metrics["normalized_progress_rate"][:] = progress_delta

        push_point_w = command.reach_target_pos_w
        contact_distance = torch.linalg.norm(
            push_point_w - ee_frame.data.target_pos_w[:, 0], dim=-1
        )
        current_goal_distance = torch.linalg.norm(
            command.goal_pos_w - target.data.root_pos_w, dim=-1
        )
        wrist_y_error = torch.abs(
            push_point_w[:, 1] - wrist_frame.data.target_pos_w[:, 0, 1]
        )
        return pushing_target_raw_reward(
            current_goal_distance,
            progress_delta,
            contact_distance,
            wrist_y_error,
            target.data.root_lin_vel_w[:, 1],
            gate_distance,
            velocity_reward_min,
            velocity_penalty_min,
        )


def homing_reward(
    env,
    command_name: str,
    success_distance: float = 0.03,
    gate_sharpness: float = 100.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if success_distance <= 0.0 or gate_sharpness <= 0.0:
        raise ValueError("Home gate distance and sharpness must be positive.")
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    goal_distance_yz = torch.linalg.vector_norm(
        command.goal_pos_w[:, 1:] - target.data.root_pos_w[:, 1:], dim=-1
    )
    joint_error = torch.sum(
        torch.abs(
            robot.data.joint_pos[:, asset_cfg.joint_ids]
            - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=-1,
    )
    home_score = torch.exp(-0.5 * joint_error)
    home_gate = 0.5 * (
        1.0 - torch.tanh(gate_sharpness * (goal_distance_yz - success_distance))
    )
    return home_score * home_gate
