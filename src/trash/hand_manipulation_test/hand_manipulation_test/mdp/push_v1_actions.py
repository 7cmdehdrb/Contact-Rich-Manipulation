"""Push-v1 OSC with persistent translation and reset-relative orientation."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from isaaclab.utils import configclass

from ..accumulation_math import accumulated_translation_target, translation_target_error_c
from ..action_math import _normalize_quaternion, quaternion_from_rotation_vector, quaternion_multiply
from .actions import CurrentFrameOscAction, CurrentFrameOscActionCfg


class AccumulatedTranslationOscAction(CurrentFrameOscAction):
    """Accumulate position once per policy step; hold it during physics substeps.

    Translation increments use the measured C axes. Position error is bounded
    in Cartesian norm at each policy update. Rotation is a bounded residual
    around the realized reset orientation, so a zero rotation command holds
    the side-pushing pose under load. The inherited OSC dynamics and joint
    effort limits apply unchanged to the resulting absolute pose target.
    """

    cfg: "AccumulatedTranslationOscActionCfg"

    def __init__(self, cfg, env):
        if not math.isfinite(cfg.position_error_limit_m) or cfg.position_error_limit_m <= 0.0:
            raise ValueError("Position target error limit must be finite and positive")
        super().__init__(cfg, env)
        if getattr(cfg, "contact_aware_impedance", False):
            if not cfg.inertial_dynamics_decoupling or cfg.partial_inertial_dynamics_decoupling:
                raise ValueError("Contact-aware V1 OSC requires the full operational inertia")
            from .push_v1_controller import ContactCartesianOscController
            self._osc = ContactCartesianOscController(self._osc, self)
            self._osc.set_command(self._desired_c_pose_b)
        self._translation_target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reference_c_quat_b = self._desired_c_pose_b[:, 3:].clone()

    @property
    def reference_c_quat_b(self) -> torch.Tensor:
        """Realized reset orientation used by the bounded rotation residual."""

        return self._reference_c_quat_b

    @property
    def position_target_error_c_m(self) -> torch.Tensor:
        """Current-C target error in metres; newly initialized rows report zero."""

        self._compute_c_pose_and_twist()
        error = translation_target_error_c(
            self._desired_c_pose_b[:, :3], self._c_pose_b[:, :3], self._c_pose_b[:, 3:]
        )
        return torch.where(self._translation_target_initialized[:, None], error, torch.zeros_like(error))

    def process_actions_for_envs(self, actions: torch.Tensor, env_ids: Sequence[int] | torch.Tensor) -> None:
        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if index.ndim != 1 or index.numel() == 0:
            if index.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            return
        if (index < 0).any() or (index >= self.num_envs).any():
            raise IndexError("env_ids contains an out-of-range environment index")
        if actions.shape != (len(index), self.action_dim):
            raise ValueError(f"Expected OSC actions with shape {(len(index), self.action_dim)}, got {tuple(actions.shape)}.")
        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, torch.zeros_like(actions))
        normalized = sanitized.clamp(-1.0, 1.0)
        self._raw_actions[index] = normalized
        invalid_action = ~finite.all(dim=-1)
        self._invalid_action[index] |= invalid_action
        self._torque_saturated[index] |= invalid_action | torch.any(sanitized != normalized, dim=-1)
        self._processed_actions[index, :3] = normalized[:, :3] * self._translation_scale
        self._processed_actions[index, 3:] = normalized[:, 3:] * self._rotation_scale

        self._compute_c_pose_and_twist()
        current = self._c_pose_b[index]
        self._reference_c_quat_b[index] = torch.where(
            self._translation_target_initialized[index, None], self._reference_c_quat_b[index], current[:, 3:]
        )
        previous_position = torch.where(
            self._translation_target_initialized[index, None], self._desired_c_pose_b[index, :3], current[:, :3]
        )
        self._desired_c_pose_b[index, :3] = accumulated_translation_target(
            previous_position, current[:, :3], current[:, 3:], self._processed_actions[index, :3],
            self.cfg.position_error_limit_m,
        )
        target_quaternion = _normalize_quaternion(
            quaternion_multiply(
                self._reference_c_quat_b[index], quaternion_from_rotation_vector(self._processed_actions[index, 3:])
            )
        )
        self._desired_c_pose_b[index, 3:] = target_quaternion
        self._translation_target_initialized[index] = True
        self._osc.set_command(self._desired_c_pose_b)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        index = slice(None) if env_ids is None or isinstance(env_ids, slice) else torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        self._translation_target_initialized[index] = True
        self._reference_c_quat_b[index] = self._desired_c_pose_b[index, 3:]
        reset_contact = getattr(self._osc, "reset_contact_history", None)
        if reset_contact is not None:
            reset_contact(index)


@configclass
class AccumulatedTranslationOscActionCfg(CurrentFrameOscActionCfg):
    class_type: type = AccumulatedTranslationOscAction
    motion_stiffness: tuple[float, float, float, float, float, float] = (200.0,) * 6
    position_error_limit_m: float = 0.06
    contact_aware_impedance: bool = False


def accumulated_translation_error(env) -> torch.Tensor:
    """Expose the arm's persistent Cartesian controller state to the policy."""

    return env.action_manager.get_term("arm_action").position_target_error_c_m


__all__ = ["AccumulatedTranslationOscAction", "AccumulatedTranslationOscActionCfg", "accumulated_translation_error"]
