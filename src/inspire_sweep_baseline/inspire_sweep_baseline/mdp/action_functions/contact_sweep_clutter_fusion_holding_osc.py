# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Relative OSC commands expressed in the EEF axes sampled at each policy step.

Local adaptation of the checked-in UR10 Reach OSC action.

Preserves RPY conversion, nonfinite-effort rejection and effort saturation.
Robot joints and the control point are supplied by the Holding configuration."""

from __future__ import annotations

from collections.abc import Sequence
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
import isaaclab.utils.math as math_utils
import torch as torch

class EEFRelativeOSCAction(OperationalSpaceControllerAction):
    """Use the full offset EEF pose as a frozen task frame for each command."""

    def __init__(self, cfg, env):
        if cfg.task_frame_rel_path is not None or cfg.controller_cfg.target_types != ["pose_rel"]:
            raise ValueError("EEF-relative OSC requires pose_rel and no external task frame")
        super().__init__(cfg, env)

    def _compute_task_frame_pose(self):
        # process_actions updates _ee_pose_b first, including body_offset.
        # Clone: physics substeps update _ee_pose_b, but this command's frame
        # must remain at its policy-step pose until the next action arrives.
        self._task_frame_pose_b = self._ee_pose_b.clone()


class UR10OperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC action term with RPY relative-pose actions and bounded effort commands."""

    cfg: "UR10OperationalSpaceControllerActionCfg"

    def __init__(self, cfg: "UR10OperationalSpaceControllerActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._torque_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def torque_saturated(self) -> torch.Tensor:
        return self._torque_saturated

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._joint_efforts[env_ids] = 0.0
        self._torque_saturated[env_ids] = False

    def _preprocess_actions(self, actions: torch.Tensor):
        super()._preprocess_actions(actions)

        if self._pose_rel_idx is None:
            return

        rot_idx = slice(self._pose_rel_idx + 3, self._pose_rel_idx + 6)
        delta_quat = math_utils.quat_from_euler_xyz(
            self._processed_actions[:, rot_idx.start],
            self._processed_actions[:, rot_idx.start + 1],
            self._processed_actions[:, rot_idx.start + 2],
        )
        self._processed_actions[:, rot_idx] = math_utils.axis_angle_from_quat(delta_quat)

    def apply_actions(self):
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()

        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )

        finite_efforts = torch.isfinite(self._joint_efforts).all(dim=1)
        self._joint_efforts[:] = torch.where(finite_efforts.unsqueeze(-1), self._joint_efforts, 0.0)

        effort_limits = self._asset.data.joint_effort_limits[:, self._joint_ids] * self.cfg.effort_limit_scale
        effort_limits = torch.clamp(effort_limits, min=self.cfg.minimum_effort_limit)
        clamped_efforts = torch.clamp(self._joint_efforts, min=-effort_limits, max=effort_limits)
        self._torque_saturated[:] = (~finite_efforts) | torch.any(
            torch.abs(clamped_efforts - self._joint_efforts) > self.cfg.saturation_tolerance, dim=1
        )
        self._joint_efforts[:] = clamped_efforts

        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)
