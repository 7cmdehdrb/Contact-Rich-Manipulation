"""Action terms owned by the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ArmJointPositionWithOpenGripperAction(JointPositionAction):
    """Apply a 6D arm target while holding every gripper joint open."""

    cfg: "ArmJointPositionWithOpenGripperActionCfg"

    def __init__(
        self,
        cfg: "ArmJointPositionWithOpenGripperActionCfg",
        env: "ManagerBasedEnv",
    ):
        super().__init__(cfg, env)
        self._gripper_joint_ids, self._gripper_joint_names = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=True
        )
        if len(self._gripper_joint_ids) != len(cfg.gripper_joint_names):
            raise ValueError(
                "Every configured gripper joint must resolve exactly once. "
                f"Resolved {self._gripper_joint_names}."
            )
        self._open_gripper_target = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            cfg.gripper_open_position,
            dtype=torch.float32,
            device=self.device,
        )

    def apply_actions(self) -> None:
        super().apply_actions()
        self._asset.set_joint_position_target(
            self._open_gripper_target,
            joint_ids=self._gripper_joint_ids,
        )


@configclass
class ArmJointPositionWithOpenGripperActionCfg(JointPositionActionCfg):
    class_type: type = ArmJointPositionWithOpenGripperAction
    gripper_joint_names: list[str] = MISSING
    gripper_open_position: float = 0.0
