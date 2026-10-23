"""Absolute, two-channel Inspire hand control owned by this package."""

from __future__ import annotations

from ...assets.robot import CLOSED_POSITIONS as HAND_CLOSED_POSITIONS
from ...assets.robot import HAND_JOINT_NAMES
from ...assets.robot import OPEN_POSITIONS as HAND_OPEN_POSITIONS
from isaaclab.managers import ActionTerm
import torch as torch

def hand_joint_targets(openness, open_positions, closed_positions):
    """Channel 0 opens all flexion joints; channel 1 opens thumb opposition."""
    joint_openness = openness[..., :1].expand(*openness.shape[:-1], 12).clone()
    joint_openness[..., 0] = openness[..., 1]
    return closed_positions + joint_openness * (open_positions - closed_positions)


class HandOpennessAction(ActionTerm):
    """Two commands in [0, 1], where one means the fully extended hand."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        if tuple(names) != HAND_JOINT_NAMES:
            raise ValueError(f"Unexpected Inspire hand joints: {names}")
        self._open_positions = torch.tensor(HAND_OPEN_POSITIONS, device=self.device)
        self._closed_positions = torch.tensor(HAND_CLOSED_POSITIONS, device=self.device)
        self._raw_actions = torch.ones((self.num_envs, 2), device=self.device)
        self._processed_actions = torch.ones_like(self._raw_actions)
        self._joint_targets = self._open_positions.expand(self.num_envs, -1).clone()

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(0.0, 1.0)
        self._joint_targets[:] = hand_joint_targets(
            self._processed_actions, self._open_positions, self._closed_positions
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self._raw_actions[index] = 1.0
        self._processed_actions[index] = 1.0
        self._joint_targets[index] = self._open_positions
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=env_ids
        )
