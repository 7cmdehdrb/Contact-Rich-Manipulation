"""Measured hand openness for Final v1 and later."""

import torch
from isaaclab.managers import SceneEntityCfg
from ...assets.robot import OPEN_POSITIONS, CLOSED_POSITIONS


def tactile_observation_noise(data, cfg):
    """Flip each reported tactile bit independently with probability 0.01."""
    return torch.where(torch.rand_like(data) < 0.01, 1.0 - data, data)


def wrist_wrench_observation_noise(data, cfg):
    """Add independent zero-mean noise: force 0.05 N, moment 0.005 Nm std."""
    std = data.new_tensor((0.05, 0.05, 0.05, 0.005, 0.005, 0.005))
    return data + torch.randn_like(data) * std


def previous_action_command(env):
    """Bound all eight raw command feedback channels with tanh.

    Keep the manager's raw actions intact for execution and action-rate costs.
    """
    return env.action_manager.action.tanh()


def hand_configuration(env, asset_cfg: SceneEntityCfg):
    """Project actual joint positions onto the two command channels, in [0, 1].

    Flexion is the least-squares inverse of hand_joint_targets; opposition is
    joint zero. This measures the hand, not its commanded action.
    """
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    closed = q.new_tensor(CLOSED_POSITIONS)
    span = q.new_tensor(OPEN_POSITIONS) - closed
    flexion = ((q[:, 1:] - closed[1:]) * span[1:]).sum(-1) / span[1:].square().sum()
    opposition = (q[:, 0] - closed[0]) / span[0]
    return torch.stack((flexion, opposition), -1).clamp(0.0, 1.0)
