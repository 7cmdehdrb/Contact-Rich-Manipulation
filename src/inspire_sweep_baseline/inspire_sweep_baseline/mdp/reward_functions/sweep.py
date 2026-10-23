"""Local copies of sweep shaping with physical-axis alignment and named Arm DOFs.

The cylinder link origin is its bottom: +9 cm is +4 cm relative to its
geometric center. Sensor channels are deliberately absent from these terms."""

from __future__ import annotations

from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
import torch as torch

def _target_rows(env):
    return torch.arange(env.num_envs, device=env.device), env.target_id[:, 0].long()


def _approach_position(env, z_offset):
    rows, targets = _target_rows(env)
    target = env.scene["object_collection"].data.object_pos_w[rows, targets].clone()
    target[:, 0] -= 0.02
    target[:, 1] -= env.target_width[:, 0] * torch.sign(env.sweep_dir[:, 1])
    target[:, 2] += z_offset
    return target


def reaching_target(env, z_offset=0.09):
    target = _approach_position(env, z_offset)
    eef = env.scene["ee_frame"].data.target_pos_w[:, 0]
    return torch.exp(-10.0 * torch.linalg.vector_norm(target - eef, dim=-1))


def align_eef_y_with_shelf_z(env):
    """World-expressed local axes are the rotation matrix columns."""
    eef = env.scene["ee_frame"].data.target_quat_w[:, 0]
    shelf = env.scene["shelf"].data.default_root_state[:, 3:7]
    eef_y = matrix_from_quat(eef)[..., :, 1]
    shelf_z = matrix_from_quat(shelf)[..., :, 2]
    cosine = torch.sum(eef_y * shelf_z, dim=-1)
    return torch.sign(cosine) * cosine.square()


def pushing_target(env, command_name="target_goal_pos", z_offset=0.09, eef_distance_threshold=0.04):
    """Original 4 cm EEF/wrist gates and piecewise object-speed shaping."""
    rows, targets = _target_rows(env)
    objects = env.scene["object_collection"]
    position = objects.data.object_pos_w[rows, targets]
    velocity = objects.data.object_lin_vel_w[rows, targets]
    goal = env.command_manager.get_command(command_name)[:, :3]
    eef = env.scene["ee_frame"].data.target_pos_w[:, 0]
    wrist = env.scene["wrist_frame"].data.target_pos_w[:, 0]
    approach = _approach_position(env, z_offset)
    distance = torch.linalg.vector_norm(goal - position, dim=-1)
    gate = (
        (torch.linalg.vector_norm(approach - eef, dim=-1) < eef_distance_threshold)
        & (torch.abs(approach[:, 1] - wrist[:, 1]) < 0.04)
    ).to(position.dtype)
    speed_y = torch.abs(velocity[:, 1])
    velocity_reward = torch.where(
        speed_y > 0.05,
        torch.where(speed_y < 0.1, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * torch.exp(-5.0 * distance),
        gate * ((1.0 - distance / 0.18) + velocity_reward),
    )


def homing_reward(env, command_name="target_goal_pos", asset_cfg=SceneEntityCfg("robot")):
    rows, targets = _target_rows(env)
    position = env.scene["object_collection"].data.object_pos_w[rows, targets]
    goal = env.command_manager.get_command(command_name)[:, :3]
    distance = torch.linalg.vector_norm(goal[:, 1:] - position[:, 1:], dim=-1)
    robot = env.scene[asset_cfg.name]
    # Preserve the source first-five Arm homing formula without relying on USD order.
    joints = robot.data.joint_pos[:, asset_cfg.joint_ids][:, :5]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_ids][:, :5]
    home = torch.exp(-0.5 * torch.sum(torch.abs(joints - default), dim=1))
    gate = 0.5 * (1.0 - torch.tanh(100.0 * (distance - 0.03)))
    return home * gate


def object_collision(env):
    """Non-target object motion penalty; naturally zero for the single object."""
    rows, targets = _target_rows(env)
    velocity = torch.round(env.scene["object_collection"].data.object_lin_vel_w.clone(), decimals=2)
    velocity[rows, targets] = 0.0
    return torch.tanh(torch.sum(torch.abs(velocity), dim=(1, 2)))


def joint_vel_l2(env, asset_cfg=SceneEntityCfg("robot")):
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(velocity.square(), dim=1)


def action_rate_l2(env):
    return torch.sum((env.action_manager.action - env.action_manager.prev_action).square(), dim=1)


class ShelfCollisionPenalty(ManagerTermBase):
    """Original shelf motion and virtual-finger/wrist proximity penalty."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._eef = env.scene["ee_frame"]
        self._fingers = env.scene["finger_frame"]
        self._wrist = env.scene["wrist_frame"]
        self._shelf = env.scene["shelf"]
        self._initial_shelf_pos = self._shelf.data.default_root_state[:, :3] + env.scene.env_origins

    def __call__(self, env):
        shelf = self._shelf.data
        moved = (
            torch.linalg.vector_norm(shelf.root_pos_w - self._initial_shelf_pos, dim=-1)
            + torch.linalg.vector_norm(shelf.root_vel_w, dim=-1)
        ) > 0.005
        surface = shelf.root_pos_w.clone()
        surface[:, 2] += 1.06
        near = torch.linalg.vector_norm(surface - self._eef.data.target_pos_w[:, 0], dim=-1) < 0.2
        finger_height = self._fingers.data.target_pos_w[:, :2, 2] - surface[:, 2, None]
        wrist_height = self._wrist.data.target_pos_w[:, 0, 2] - surface[:, 2]
        finger_penalty = (1.0 - finger_height / 0.02).clamp(0.0, 1.0).sum(dim=1)
        wrist_penalty = (1.0 - wrist_height / 0.08).clamp(0.0, 1.0)
        return moved.to(surface.dtype) + near * (finger_penalty + wrist_penalty)
