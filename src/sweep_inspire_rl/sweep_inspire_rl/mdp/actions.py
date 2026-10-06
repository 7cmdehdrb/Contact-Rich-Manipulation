"""Reference OSC with a fixed palmar orientation and nearly open Inspire hand."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass

from hand_manipulation_rl.mdp.actions import (
    InspireHandSynergyAction,
    InspireHandSynergyActionCfg,
    inspire_joint_positions_to_synergy,
    inspire_synergy_to_joint_positions,
    shift_jacobian_to_point,
)


# H +X points up, H +Y points toward shelf right (+world Y), and the fingers
# (H +Z) point into the shelf (-world X). Quaternions are always wxyz.
RIGHT_PALM_QUAT_WXYZ = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)


def bounded_hand_openness(actions: torch.Tensor, min_openness: float = 0.8) -> torch.Tensor:
    """Map two unit-interval policy coordinates to physical openness."""

    if actions.ndim != 2 or actions.shape[-1] != 2:
        raise ValueError("Hand actions must have shape (num_envs, 2)")
    if not math.isfinite(min_openness) or not 0.0 <= min_openness < 1.0:
        raise ValueError("min_openness must lie in [0, 1)")
    unit_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return min_openness + (1.0 - min_openness) * unit_actions


def orientation_delta_to_target(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Return the shortest left-multiplied rotation vector to a target."""

    delta_quat = math_utils.quat_mul(target_quat, math_utils.quat_inv(current_quat))
    return math_utils.axis_angle_from_quat(math_utils.quat_unique(delta_quat))


class RightPalmOscAction(OperationalSpaceControllerAction):
    """Keep the source 6D pose-relative OSC while holding the right-facing palm.

    Translation retains the reference action scales. The three rotation
    inputs remain in the policy interface, but their executed target is the
    fixed palmar quaternion rather than an independently rotatable hand.
    COM-based PhysX Jacobians and link-origin velocities are shifted to the
    same configured control point before the source OSC computes efforts.
    """

    cfg: "RightPalmOscActionCfg"

    def __init__(self, cfg, env):
        if tuple(cfg.controller_cfg.target_types) != ("pose_rel",):
            raise ValueError("Right-palm OSC requires exactly one pose_rel target")
        if cfg.controller_cfg.impedance_mode != "fixed":
            raise ValueError("Right-palm OSC requires fixed impedance")
        if cfg.body_name != "inspire_base_link":
            raise ValueError("Right-palm OSC must control inspire_base_link")
        super().__init__(cfg, env)
        if not self._asset.is_fixed_base:
            raise ValueError("Right-palm OSC requires a fixed robot base")

    def _preprocess_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 6):
            raise ValueError(f"Expected OSC actions with shape {(self.num_envs, 6)}")
        super()._preprocess_actions(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0))
        hand_quat_w = torch.tensor(
            self.cfg.palm_quat_w, dtype=actions.dtype, device=self.device
        ).expand(self.num_envs, -1)
        ee_quat_w = hand_quat_w
        if self._offset_rot is not None:
            ee_quat_w = math_utils.quat_mul(hand_quat_w, self._offset_rot)
        desired_quat_b = math_utils.quat_mul(
            math_utils.quat_inv(self._asset.data.root_quat_w), ee_quat_w
        )
        current_quat = self._ee_pose_b[:, 3:7]
        if self._task_frame_pose_b is not None:
            task_quat_inv = math_utils.quat_inv(self._task_frame_pose_b[:, 3:7])
            current_quat = math_utils.quat_mul(task_quat_inv, current_quat)
            desired_quat_b = math_utils.quat_mul(task_quat_inv, desired_quat_b)
        self._processed_actions[:, 3:6] = orientation_delta_to_target(current_quat, desired_quat_b)

    def _compute_ee_jacobian(self) -> None:
        # Parent OSC uses a link-origin offset directly against a COM Jacobian.
        # Compute the physical COM-to-C displacement in root coordinates.
        self._compute_ee_pose()
        jacobian_com_b = self.jacobian_b
        com_pos_b = math_utils.quat_apply_inverse(
            self._asset.data.root_quat_w,
            self._asset.data.body_com_pos_w[:, self._ee_body_idx] - self._asset.data.root_pos_w,
        )
        com_to_c_b = self._ee_pose_b[:, :3] - com_pos_b
        self._jacobian_b[:] = shift_jacobian_to_point(jacobian_com_b, com_to_c_b)

    def _compute_ee_velocity(self) -> None:
        # The legacy body_vel_w alias is at the COM. Read the link-origin
        # velocity explicitly, then transport it to C, matching its Jacobian.
        data = self._asset.data
        self._ee_vel_w[:] = data.body_link_vel_w[:, self._ee_body_idx]
        relative_vel_w = self._ee_vel_w - data.root_link_vel_w
        self._ee_vel_b[:, :3] = math_utils.quat_apply_inverse(data.root_quat_w, relative_vel_w[:, :3])
        self._ee_vel_b[:, 3:] = math_utils.quat_apply_inverse(data.root_quat_w, relative_vel_w[:, 3:])
        if self._offset_pos is not None:
            h_to_c_b = math_utils.quat_apply(self._ee_pose_b_no_offset[:, 3:7], self._offset_pos)
            self._ee_vel_b[:, :3] += torch.linalg.cross(self._ee_vel_b[:, 3:], h_to_c_b, dim=-1)


@configclass
class RightPalmOscActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = RightPalmOscAction
    palm_quat_w: tuple[float, float, float, float] = RIGHT_PALM_QUAT_WXYZ


class NearlyOpenHandSynergyAction(InspireHandSynergyAction):
    """Unit-interval 2D actions using the source rate limits and follower map."""

    cfg: "NearlyOpenHandSynergyActionCfg"

    def __init__(self, cfg, env):
        bounded_hand_openness(torch.zeros(1, 2), cfg.min_openness)
        super().__init__(cfg, env)
        self.reset()

    @property
    def raw_actions(self) -> torch.Tensor:
        """Sanitized policy coordinates in [0, 1]."""

        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Executed target openness in [min_openness, 1]."""

        return self._processed_actions

    def process_actions_for_envs(self, actions, env_ids) -> None:
        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        desired_openness = bounded_hand_openness(actions, self.cfg.min_openness)
        # Reuse the source controller's synergy and per-joint rate limiting.
        super().process_actions_for_envs(2.0 * desired_openness - 1.0, index)
        self._raw_actions[index] = torch.nan_to_num(
            actions, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        self._processed_actions[index] = self._synergy_target[index]
        self._action_saturated[index] |= (~torch.isfinite(actions).all(dim=-1)) | (
            actions != self._raw_actions[index]
        ).any(dim=-1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        # The source reset interprets any slice as a full reset. Resolve a
        # slice here to preserve other running vector environments.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super().reset(env_ids)
        if env_ids is None:
            index, sim_env_ids = slice(None), None
        else:
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            sim_env_ids = index
        # Reset events put the actual hand in this range. Clamp position targets
        # too, so an external reset cannot command a closed hand accidentally.
        openness = self._synergy_target[index].clamp(self.cfg.min_openness, 1.0)
        self._joint_targets[index] = inspire_synergy_to_joint_positions(
            openness, self._open_positions, self._closed_positions
        )
        self._synergy_target[index] = inspire_joint_positions_to_synergy(
            self._joint_targets[index], self._open_positions, self._closed_positions
        )
        self._processed_actions[index] = self._synergy_target[index]
        self._raw_actions[index] = (
            (self._synergy_target[index] - self.cfg.min_openness) / (1.0 - self.cfg.min_openness)
        ).clamp(0.0, 1.0)
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=sim_env_ids
        )


@configclass
class NearlyOpenHandSynergyActionCfg(InspireHandSynergyActionCfg):
    class_type: type = NearlyOpenHandSynergyAction
    min_openness: float = 0.8


__all__ = [
    "NearlyOpenHandSynergyAction",
    "NearlyOpenHandSynergyActionCfg",
    "RIGHT_PALM_QUAT_WXYZ",
    "RightPalmOscAction",
    "RightPalmOscActionCfg",
    "bounded_hand_openness",
    "orientation_delta_to_target",
]
