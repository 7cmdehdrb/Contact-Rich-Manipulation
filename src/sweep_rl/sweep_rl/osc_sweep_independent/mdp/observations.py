"""Policy observations for the standalone sweep task."""

from __future__ import annotations

from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .common import pose_w_to_root_rpy, target_contact_data_w


def end_effector_pose_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return pose_w_to_root_rpy(
        robot,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )


def virtual_ft_wrench_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return -robot.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids[0], :]


def target_contact_point_b(
    env,
    robot_cfg: SceneEntityCfg,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    point_w, _, contact_mask = target_contact_data_w(
        env, sensor_names=sensor_names, force_threshold=force_threshold
    )
    point_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, point_w - robot.data.root_pos_w
    )
    return torch.where(contact_mask.unsqueeze(-1), point_b, torch.zeros_like(point_b))


def initial_target_pose_b(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).initial_pose_b


def task_phase(env, command_name: str) -> torch.Tensor:
    phase = env.command_manager.get_term(command_name).task_phase
    return phase.float().unsqueeze(-1)


def vector_uniform_noise(
    data: torch.Tensor, cfg: "VectorUniformNoiseCfg"
) -> torch.Tensor:
    """Add per-component noise from Hydra-serializable list bounds."""
    n_min = torch.as_tensor(cfg.n_min, device=data.device, dtype=data.dtype)
    n_max = torch.as_tensor(cfg.n_max, device=data.device, dtype=data.dtype)
    if n_min.shape != data.shape[1:] or n_max.shape != data.shape[1:]:
        raise ValueError(
            "Vector noise bounds must match one observation row: "
            f"data={tuple(data.shape[1:])}, n_min={tuple(n_min.shape)}, "
            f"n_max={tuple(n_max.shape)}."
        )
    return data + torch.rand_like(data) * (n_max - n_min) + n_min


@configclass
class VectorUniformNoiseCfg(NoiseCfg):
    """Per-component additive noise with Hydra-safe configuration values."""

    func = vector_uniform_noise
    n_min: list[float] = MISSING
    n_max: list[float] = MISSING


def masked_uniform_noise(data: torch.Tensor, cfg: "MaskedUniformNoiseCfg") -> torch.Tensor:
    """Add component-wise noise while keeping no-contact zero rows exactly zero."""
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)
    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    no_measurement = torch.all(data == 0.0, dim=-1, keepdim=True)
    noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    return torch.where(no_measurement, data, data + noise)


def componentwise_uniform_noise(
    data: torch.Tensor, cfg: "ComponentwiseUniformNoiseCfg"
) -> torch.Tensor:
    """Add uniform noise with Hydra-compatible per-component bounds."""
    n_min = torch.as_tensor(cfg.n_min, device=data.device, dtype=data.dtype)
    n_max = torch.as_tensor(cfg.n_max, device=data.device, dtype=data.dtype)
    return data + torch.rand_like(data) * (n_max - n_min) + n_min


@configclass
class ComponentwiseUniformNoiseCfg(NoiseCfg):
    """Uniform additive noise whose bounds remain serializable by Hydra."""

    func = componentwise_uniform_noise
    n_min: tuple[float, ...] = (-1.0,)
    n_max: tuple[float, ...] = (1.0,)


@configclass
class MaskedUniformNoiseCfg(NoiseCfg):
    """Uniform additive noise that preserves the contact-point sentinel."""

    func = masked_uniform_noise
    n_min: torch.Tensor | float = -1.0
    n_max: torch.Tensor | float = 1.0
