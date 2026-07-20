"""Randomized 12-D variable-stiffness OSC with an always-open gripper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class IndependentSweepOscAction(OperationalSpaceControllerAction):
    """Map ``[stiffness(6), temporal relative pose(6)]`` to OSC torque."""

    cfg: "IndependentSweepOscActionCfg"

    def __init__(self, cfg: "IndependentSweepOscActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self.action_dim != 12:
            raise ValueError(f"Expected a 12-D OSC action, received {self.action_dim} dimensions.")
        self._torque_saturated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stiffness_command = torch.zeros(self.num_envs, 6, device=self.device)
        self._pose_command = torch.zeros(self.num_envs, 6, device=self.device)
        self._stiffness_calibration = torch.ones(self.num_envs, 1, device=self.device)
        self._damping_calibration = torch.ones(self.num_envs, 1, device=self.device)
        self._effort_calibration = torch.ones(self.num_envs, 1, device=self.device)

        self._gripper_joint_ids, _ = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=True
        )
        if len(self._gripper_joint_ids) == 0:
            raise ValueError("No gripper joints matched gripper_joint_names.")
        self._gripper_open_targets = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            cfg.gripper_open_position,
            device=self.device,
        )

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    @property
    def torque_saturated(self) -> torch.Tensor:
        return self._torque_saturated

    @property
    def stiffness_command(self) -> torch.Tensor:
        return self._stiffness_command

    @property
    def pose_command(self) -> torch.Tensor:
        return self._pose_command

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._joint_efforts[env_ids] = 0.0
        self._torque_saturated[env_ids] = False
        self._stiffness_command[env_ids] = 0.0
        self._pose_command[env_ids] = 0.0
        selected_shape = self._stiffness_calibration[env_ids].shape
        self._stiffness_calibration[env_ids] = torch.empty(
            selected_shape, device=self.device
        ).uniform_(*self.cfg.stiffness_calibration_range)
        self._damping_calibration[env_ids] = torch.empty(
            selected_shape, device=self.device
        ).uniform_(*self.cfg.damping_calibration_range)
        self._effort_calibration[env_ids] = torch.empty(
            selected_shape, device=self.device
        ).uniform_(*self.cfg.effort_calibration_range)
        self._asset.set_joint_position_target(
            self._gripper_open_targets[env_ids],
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )

    def _preprocess_actions(self, actions: torch.Tensor) -> None:
        expected_shape = (self.num_envs, 12)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected actions with shape {expected_shape}, got {tuple(actions.shape)}.")

        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, 0.0)
        normalized = torch.clamp(sanitized, -1.0, 1.0)
        self._raw_actions[:] = sanitized

        stiffness_min, stiffness_max = self.cfg.controller_cfg.motion_stiffness_limits_task
        stiffness = (
            0.5 * (normalized[:, :6] + 1.0) * (stiffness_max - stiffness_min)
            + stiffness_min
        )
        stiffness *= self._stiffness_calibration
        stiffness.clamp_(min=stiffness_min, max=stiffness_max)

        pose_command = normalized[:, 6:12].clone()
        pose_command[:, :3] *= self._position_scale
        pose_command[:, 3:6] *= self._orientation_scale
        delta_quaternion = math_utils.quat_from_euler_xyz(
            pose_command[:, 3], pose_command[:, 4], pose_command[:, 5]
        )
        pose_command[:, 3:6] = math_utils.axis_angle_from_quat(delta_quaternion)

        self._processed_actions[:, :6] = pose_command
        self._processed_actions[:, 6:12] = stiffness
        self._stiffness_command[:] = stiffness
        self._pose_command[:] = pose_command
        clipped = torch.any(
            torch.abs(sanitized - normalized) > self.cfg.saturation_tolerance, dim=1
        )
        self._torque_saturated[:] = (~finite.all(dim=1)) | clipped

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        # Isaac Lab derives damping from commanded stiffness. A small per-episode
        # multiplier models OSC gain calibration error without increasing action size.
        self._osc._motion_d_gains_task *= self._damping_calibration.unsqueeze(-1)

    def apply_actions(self) -> None:
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
        safe_efforts = torch.where(
            finite_efforts.unsqueeze(-1), self._joint_efforts, 0.0
        )
        effort_limits = (
            self._asset.data.joint_effort_limits[:, self._joint_ids]
            * self.cfg.effort_limit_scale
            * self._effort_calibration
        )
        effort_limits.clamp_(min=self.cfg.minimum_effort_limit)
        clamped = torch.clamp(safe_efforts, min=-effort_limits, max=effort_limits)
        effort_clipped = torch.any(
            torch.abs(clamped - safe_efforts) > self.cfg.saturation_tolerance, dim=1
        )
        self._torque_saturated |= (~finite_efforts) | effort_clipped
        self._joint_efforts[:] = clamped
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(
            self._gripper_open_targets, joint_ids=self._gripper_joint_ids
        )


@configclass
class IndependentSweepOscActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = IndependentSweepOscAction
    effort_limit_scale: float = 0.9
    minimum_effort_limit: float = 1.0e-6
    saturation_tolerance: float = 1.0e-6
    stiffness_calibration_range: tuple[float, float] = (0.95, 1.05)
    damping_calibration_range: tuple[float, float] = (0.95, 1.05)
    effort_calibration_range: tuple[float, float] = (0.97, 1.03)
    gripper_joint_names: list[str] = [".*(finger|knuckle).*"]
    gripper_open_position: float = 0.0
