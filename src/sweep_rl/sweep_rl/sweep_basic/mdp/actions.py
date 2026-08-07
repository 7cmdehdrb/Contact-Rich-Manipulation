"""Six-dimensional arm control with an always-open gripper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .action_math import incremental_joint_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class OpenGripperRelativeJointPositionAction(JointPositionAction):
    """Apply latched current-relative arm targets and hold the gripper open."""

    cfg: "OpenGripperRelativeJointPositionActionCfg"

    def __init__(
        self, cfg: "OpenGripperRelativeJointPositionActionCfg", env: "ManagerBasedEnv"
    ):
        super().__init__(cfg, env)
        self._gripper_joint_ids, _ = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=True
        )
        if len(self._gripper_joint_ids) == 0:
            raise ValueError("No gripper joints matched gripper_joint_names.")
        self._gripper_open_targets = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            cfg.gripper_open_position,
            dtype=torch.float32,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Latch ``q_current + scale * action`` once per control step."""
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        normalized_action, target = incremental_joint_target(
            joint_pos,
            actions,
            joint_limits,
            scale=self._scale,
        )
        self._raw_actions[:] = normalized_action
        self._processed_actions[:] = target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = self._asset.data.joint_pos[env_ids][
            :, self._joint_ids
        ]
        self._asset.set_joint_position_target(
            self._gripper_open_targets[env_ids],
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )

    def apply_actions(self) -> None:
        super().apply_actions()
        self._asset.set_joint_position_target(
            self._gripper_open_targets, joint_ids=self._gripper_joint_ids
        )


@configclass
class OpenGripperRelativeJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = OpenGripperRelativeJointPositionAction
    use_default_offset: bool = False
    gripper_joint_names: list[str] = [".*(finger|knuckle).*"]
    gripper_open_position: float = 0.0
