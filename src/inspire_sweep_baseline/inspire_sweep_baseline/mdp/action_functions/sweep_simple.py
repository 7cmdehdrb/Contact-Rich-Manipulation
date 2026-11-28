"""Independent six-drive absolute joint position control for Simple Finger."""
import torch
from isaaclab.managers import ActionTerm

from ...assets.robot import HAND_JOINT_NAMES
from ...assets.hand_kinematics import (
    HAND_DRIVE_UPPER_LIMITS, HAND_MIMIC_DRIVE_INDICES, HAND_MIMIC_MULTIPLIERS,
)


class FingerJointPositionAction(ActionTerm):
    """Raw radians, clamped to URDF limits; follower targets preserve mimic ratios."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        if tuple(names) != HAND_JOINT_NAMES:
            raise ValueError(f'Unexpected Inspire hand joints: {names}')
        self._upper_limits = torch.tensor(HAND_DRIVE_UPPER_LIMITS, device=self.device)
        self._drive_indices = torch.tensor(HAND_MIMIC_DRIVE_INDICES, device=self.device)
        self._multipliers = torch.tensor(HAND_MIMIC_MULTIPLIERS, device=self.device)
        self._raw_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_targets = torch.zeros((self.num_envs, 12), device=self.device)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.minimum(actions.clamp_min(0.), self._upper_limits)
        self._joint_targets[:] = self._processed_actions[:, self._drive_indices] * self._multipliers

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self._raw_actions[index] = 0.
        self._processed_actions[index] = 0.
        self._joint_targets[index] = 0.
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=env_ids,
        )


class BoundedFingerJointPositionAction(FingerJointPositionAction):
    def process_actions(self, actions):
        # Bound the term's raw history too, before the existing follower mapping.
        super().process_actions(torch.minimum(torch.nan_to_num(actions).clamp_min(0.), self._upper_limits))


class NormalizedFingerJointPositionAction(FingerJointPositionAction):
    """Six normalized commands: 0 is open, 1 is each drive's URDF upper limit.

    Keep raw commands in policy units for diagnostics and penalties. Processed
    commands and follower targets retain the parent's physical-radian units.
    """

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(0.0, 1.0) * self._upper_limits
        self._joint_targets[:] = self._processed_actions[:, self._drive_indices] * self._multipliers

    def initialize_hold(self, actions, env_ids):
        """Child reset only: match normalized history and physical hold targets."""
        self._raw_actions[env_ids] = actions
        processed = actions.clamp(0., 1.)*self._upper_limits
        targets = processed[:, self._drive_indices]*self._multipliers
        self._processed_actions[env_ids] = processed
        self._joint_targets[env_ids] = targets
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids, env_ids=env_ids)
