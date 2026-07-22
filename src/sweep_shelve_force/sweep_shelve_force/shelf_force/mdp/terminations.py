"""Failure conditions matching the random shelf-sweep example."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

from .common import selected_object_state, virtual_ft_wrench_b


def object_drop_or_flip(
    env,
    command_name: str,
    minimum_height: float,
    maximum_tilt: float,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    objects: RigidObjectCollection = env.scene[command.cfg.asset_name]
    state = objects.data.object_link_state_w
    dropped = state[:, :, 2] < minimum_height
    roll, pitch, _ = euler_xyz_from_quat(state[:, :, 3:7].reshape(-1, 4))
    roll = torch.atan2(torch.sin(roll), torch.cos(roll)).view(env.num_envs, -1)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)).view(env.num_envs, -1)
    flipped = (torch.abs(roll) > maximum_tilt) | (torch.abs(pitch) > maximum_tilt)
    return torch.any(dropped | flipped, dim=1)


def target_too_fast(env, command_name: str, speed_limit: float) -> torch.Tensor:
    speed = torch.linalg.norm(selected_object_state(env, command_name)[:, 7:10], dim=-1)
    return speed > speed_limit


def shelf_collision(
    env,
    shelf_cfg,
    finger_frame_cfg,
    wrist_frame_cfg,
    velocity_threshold: float,
) -> torch.Tensor:
    shelf: RigidObject = env.scene[shelf_cfg.name]
    fingers: FrameTransformer = env.scene[finger_frame_cfg.name]
    wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
    shelf_top_z = shelf.data.root_pos_w[:, 2] + 1.05
    left_gap = fingers.data.target_pos_w[:, 0, 2] - shelf_top_z
    right_gap = fingers.data.target_pos_w[:, 1, 2] - shelf_top_z
    wrist_gap = wrist.data.target_pos_w[:, 0, 2] - shelf_top_z
    return (
        (torch.linalg.norm(shelf.data.root_vel_w, dim=-1) > velocity_threshold)
        | (left_gap < 0.01)
        | (right_gap < 0.01)
        | (wrist_gap < 0.07)
    )


def arm_joint_speed(env, asset_cfg, speed_limit: float) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > speed_limit,
        dim=-1,
    )


def excessive_wrench(
    env,
    ft_cfg,
    force_limit: float,
    torque_limit: float,
) -> torch.Tensor:
    """Terminate when the inline sensor exceeds its Cartesian safety limits."""
    wrench_b = virtual_ft_wrench_b(env, ft_cfg)
    force = torch.linalg.norm(wrench_b[:, :3], dim=-1)
    torque = torch.linalg.norm(wrench_b[:, 3:], dim=-1)
    return (force > force_limit) | (torque > torque_limit)
