"""Shared tensors for the independent shelf-force MDP."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection


def selected_object_state(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    objects: RigidObjectCollection = env.scene[command.cfg.asset_name]
    env_ids = torch.arange(env.num_envs, device=env.device)
    return objects.data.object_state_w[env_ids, command.target_ids]


def selected_object_pos_w(env, command_name: str) -> torch.Tensor:
    return selected_object_state(env, command_name)[:, :3]


def virtual_ft_wrench_b(env, asset_cfg) -> torch.Tensor:
    """Return the inline-joint wrench expressed in robot-base axes."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    wrench_sensor = -robot.data.body_incoming_joint_wrench_b[:, body_id, :]
    _, sensor_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    force_b = math_utils.quat_apply(sensor_quat_b, wrench_sensor[:, :3])
    torque_b = math_utils.quat_apply(sensor_quat_b, wrench_sensor[:, 3:])
    return torch.cat((force_b, torque_b), dim=-1)
