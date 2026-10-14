"""Manager action terms for current-EEF OSC and two-coordinate Inspire control."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..action_math import (
    INSPIRE_CLOSED_JOINT_POSITIONS,
    INSPIRE_HAND_JOINT_NAMES,
    INSPIRE_OPEN_JOINT_POSITIONS,
    UR5E_ARM_JOINT_NAMES,
    UR5E_EFFORT_LIMITS_NM,
    _normalize_quaternion,
    com_to_target_offset,
    compose_fixed_offset_pose,
    current_frame_delta_target,
    inspire_joint_positions_to_synergy,
    inspire_synergy_to_joint_positions,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_matrix,
    shift_jacobian_to_point,
    shift_twist_to_point,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _as_six_vector(value: Sequence[float], name: str, device: str) -> torch.Tensor:
    if len(value) != 6:
        raise ValueError(f"{name} must contain six values, got {len(value)}.")
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_hand_vector(value: Sequence[float], name: str, device: str) -> torch.Tensor:
    if len(value) != len(INSPIRE_HAND_JOINT_NAMES):
        raise ValueError(f"{name} must contain twelve values, got {len(value)}.")
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


class CurrentFrameOscAction(ActionTerm):
    """Six-dimensional fixed-gain OSC action about a virtual frame C.

    ``body_name`` identifies the physical parent frame H. ``body_offset_*`` is
    the fixed transform ``T_HC``.  Each policy action is clipped to ``[-1, 1]``,
    scaled to metres and radians, and applied to the measured C pose exactly
    once.  The resulting absolute target is held across simulation substeps.
    """

    cfg: "CurrentFrameOscActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "CurrentFrameOscActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"{self.__class__.__name__} requires an Articulation asset.")
        if not self._asset.is_fixed_base:
            raise ValueError("Reaching OSC requires the configured fixed-base UR5e articulation.")

        self._joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(joint_names) != tuple(cfg.joint_names) or tuple(joint_names) != UR5E_ARM_JOINT_NAMES:
            raise ValueError(
                "OSC arm joints must resolve exactly in UR5e kinematic order; "
                f"configured={cfg.joint_names}, resolved={joint_names}."
            )
        self._joint_ids_tensor = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)

        body_ids, body_names = self._asset.find_bodies(cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"body_name must resolve exactly one H body, got {body_names}.")
        self._body_idx = int(body_ids[0])
        if self._body_idx == 0:
            raise ValueError("The fixed-base root body has no PhysX Jacobian row and cannot be H.")
        self._jacobian_body_idx = self._body_idx - 1

        action_scale = _as_six_vector(
            (*cfg.translation_scale, *cfg.rotation_scale), "action scales", self.device
        )
        self._translation_scale = action_scale[:3]
        self._rotation_scale = action_scale[3:]
        if torch.any(self._translation_scale <= 0.0) or torch.any(self._rotation_scale <= 0.0):
            raise ValueError("Translation and rotation action scales must be positive.")

        offset_position_h = torch.as_tensor(cfg.body_offset_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
        offset_quaternion_h = torch.as_tensor(
            cfg.body_offset_quat, dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        if not torch.isfinite(offset_position_h).all() or not torch.isfinite(offset_quaternion_h).all():
            raise ValueError("The fixed H-to-C transform must contain only finite values.")
        if torch.linalg.vector_norm(offset_quaternion_h).item() <= 1.0e-12:
            raise ValueError("body_offset_quat cannot be a zero quaternion.")
        self._offset_position_h = offset_position_h.repeat(self.num_envs, 1)
        self._offset_quaternion_h = _normalize_quaternion(offset_quaternion_h).repeat(self.num_envs, 1)

        if len(cfg.motion_stiffness) != 6 or len(cfg.motion_damping_ratio) != 6:
            raise ValueError("OSC stiffness and damping-ratio configurations must contain six values each.")
        if any(value < 0.0 for value in cfg.motion_stiffness):
            raise ValueError("OSC stiffness values must be non-negative.")
        if any(value < 0.0 for value in cfg.motion_damping_ratio):
            raise ValueError("OSC damping ratios must be non-negative.")
        controller_cfg = OperationalSpaceControllerCfg(
            target_types=("pose_abs",),
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
            inertial_dynamics_decoupling=cfg.inertial_dynamics_decoupling,
            partial_inertial_dynamics_decoupling=cfg.partial_inertial_dynamics_decoupling,
            gravity_compensation=cfg.gravity_compensation,
            impedance_mode="fixed",
            motion_stiffness_task=cfg.motion_stiffness,
            motion_damping_ratio_task=cfg.motion_damping_ratio,
            nullspace_control="none",
        )
        self._osc = OperationalSpaceController(controller_cfg, self.num_envs, self.device)

        self._configured_effort_limits = _as_six_vector(cfg.effort_limits, "effort_limits", self.device)
        if torch.any(self._configured_effort_limits <= 0.0):
            raise ValueError("Every OSC effort limit must be positive.")
        if not 0.0 < cfg.effort_limit_scale <= 1.0:
            raise ValueError("effort_limit_scale must lie in (0, 1].")
        if not math.isfinite(cfg.saturation_tolerance) or cfg.saturation_tolerance < 0.0:
            raise ValueError("saturation_tolerance must be finite and non-negative.")
        if cfg.partial_inertial_dynamics_decoupling and not cfg.inertial_dynamics_decoupling:
            raise ValueError("Partial inertial decoupling requires inertial_dynamics_decoupling=True.")

        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._desired_c_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._desired_c_pose_b[:, 3] = 1.0
        self._c_pose_b = self._desired_c_pose_b.clone()
        self._c_twist_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._jacobian_c_twist_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._c_jacobian_b = torch.zeros(self.num_envs, 6, 6, device=self.device)
        self._joint_efforts = torch.zeros(self.num_envs, 6, device=self.device)
        self._torque_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_action = torch.zeros_like(self._torque_saturated)
        self._r_hc_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical local increments ``[metres, radians]`` held this policy step."""

        return self._processed_actions

    @property
    def desired_c_pose_b(self) -> torch.Tensor:
        return self._desired_c_pose_b

    @property
    def c_pose_b(self) -> torch.Tensor:
        """Latest actual C pose in B as ``[position, quaternion_wxyz]``."""

        return self._c_pose_b

    @property
    def c_twist_b(self) -> torch.Tensor:
        """Latest actual C link-origin twist in B."""

        return self._c_twist_b

    @property
    def c_jacobian_b(self) -> torch.Tensor:
        return self._c_jacobian_b

    @property
    def joint_ids(self) -> tuple[int, ...]:
        """Resolved arm joint indices in UR5e kinematic order."""

        return tuple(self._joint_ids)

    @property
    def jacobian_c_twist_b(self) -> torch.Tensor:
        """Latest ``J_C qdot`` sampled at the same instant as the OSC update."""

        return self._jacobian_c_twist_b

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    @property
    def torque_saturated(self) -> torch.Tensor:
        """Per-environment action/effort sanitation or clipping during this policy step."""

        return self._torque_saturated

    @property
    def invalid_action(self) -> torch.Tensor:
        return self._invalid_action

    def process_actions(self, actions: torch.Tensor) -> None:
        # Start one policy-step diagnostic window.
        self._invalid_action[:] = False
        self._torque_saturated[:] = False
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.process_actions_for_envs(actions, env_ids)

    def process_actions_for_envs(
        self, actions: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Process new commands only for selected vector environments.

        This permits selected rows to be processed without changing targets
        for the other vector environments.
        """

        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if index.ndim != 1 or index.numel() == 0:
            if index.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            return
        if (index < 0).any() or (index >= self.num_envs).any():
            raise IndexError("env_ids contains an out-of-range environment index")
        expected_shape = (len(index), self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected OSC actions with shape {expected_shape}, got {tuple(actions.shape)}.")
        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, torch.zeros_like(actions))
        normalized = sanitized.clamp(-1.0, 1.0)
        self._raw_actions[index] = normalized
        invalid_action = ~finite.all(dim=-1)
        self._invalid_action[index] |= invalid_action
        self._torque_saturated[index] |= invalid_action | torch.any(
            sanitized != normalized, dim=-1
        )

        self._processed_actions[index, :3] = normalized[:, :3] * self._translation_scale
        self._processed_actions[index, 3:] = normalized[:, 3:] * self._rotation_scale
        self._compute_c_pose_and_twist()
        target_position_b, target_quaternion_b = current_frame_delta_target(
            self._c_pose_b[index, :3],
            self._c_pose_b[index, 3:],
            self._processed_actions[index, :3],
            self._processed_actions[index, 3:],
        )
        self._desired_c_pose_b[index, :3] = target_position_b
        self._desired_c_pose_b[index, 3:] = target_quaternion_b
        self._osc.set_command(self._desired_c_pose_b)

    def apply_actions(self) -> None:
        self._compute_c_pose_and_twist()
        self._compute_c_jacobian()
        arm_joint_velocity = torch.index_select(
            self._asset.data.joint_vel, 1, self._joint_ids_tensor
        )
        self._jacobian_c_twist_b[:] = torch.bmm(
            self._c_jacobian_b, arm_joint_velocity.unsqueeze(-1)
        ).squeeze(-1)

        mass_matrix = None
        if self.cfg.inertial_dynamics_decoupling:
            mass_matrix_full = self._asset.root_physx_view.get_generalized_mass_matrices()
            mass_matrix = torch.index_select(mass_matrix_full, 1, self._joint_ids_tensor)
            mass_matrix = torch.index_select(mass_matrix, 2, self._joint_ids_tensor)
        gravity = None
        if self.cfg.gravity_compensation:
            gravity_full = self._asset.root_physx_view.get_gravity_compensation_forces()
            gravity = torch.index_select(gravity_full, 1, self._joint_ids_tensor)

        computed_efforts = self._osc.compute(
            jacobian_b=self._c_jacobian_b,
            current_ee_pose_b=self._c_pose_b,
            current_ee_vel_b=self._c_twist_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
        )
        finite_efforts = torch.isfinite(computed_efforts).all(dim=-1)
        safe_efforts = torch.where(finite_efforts.unsqueeze(-1), computed_efforts, torch.zeros_like(computed_efforts))

        asset_limits = torch.index_select(self._asset.data.joint_effort_limits, 1, self._joint_ids_tensor)
        configured_limits = self._configured_effort_limits.unsqueeze(0).expand_as(asset_limits)
        # A non-finite PhysX limit means "unbounded"; the explicit task limit
        # remains authoritative in that case.
        finite_positive_asset_limit = torch.isfinite(asset_limits) & (asset_limits > 0.0)
        effective_limits = torch.where(
            finite_positive_asset_limit,
            torch.minimum(asset_limits, configured_limits),
            configured_limits,
        ) * self.cfg.effort_limit_scale
        clamped_efforts = torch.clamp(safe_efforts, min=-effective_limits, max=effective_limits)
        clipped = torch.any(torch.abs(clamped_efforts - safe_efforts) > self.cfg.saturation_tolerance, dim=-1)
        self._torque_saturated |= (~finite_efforts) | clipped
        self._joint_efforts[:] = clamped_efforts
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            index: slice | torch.Tensor = slice(None)
        else:
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._raw_actions[index] = 0.0
        self._processed_actions[index] = 0.0
        self._joint_efforts[index] = 0.0
        self._torque_saturated[index] = False
        self._invalid_action[index] = False

        # Do not call OperationalSpaceController.reset(): it has no partial
        # reset and would discard targets for environments still running.
        self._compute_c_pose_and_twist()
        self._desired_c_pose_b[index] = self._c_pose_b[index]
        self._osc.set_command(self._desired_c_pose_b)

    def reset_to_current(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Re-latch the measured C pose after reset events and FK finalization."""

        self.reset(env_ids)

    def _compute_c_pose_and_twist(self) -> None:
        root_position_w = self._asset.data.root_link_pos_w
        root_quaternion_w = _normalize_quaternion(self._asset.data.root_link_quat_w)
        root_twist_w = self._asset.data.root_link_vel_w
        body_position_w = self._asset.data.body_link_pos_w[:, self._body_idx]
        body_quaternion_w = _normalize_quaternion(self._asset.data.body_link_quat_w[:, self._body_idx])
        body_twist_w = self._asset.data.body_link_vel_w[:, self._body_idx]

        quaternion_bw = quaternion_conjugate(root_quaternion_w)
        position_h_b = quaternion_rotate(quaternion_bw, body_position_w - root_position_w)
        quaternion_h_b = _normalize_quaternion(quaternion_multiply(quaternion_bw, body_quaternion_w))
        position_c_b, quaternion_c_b, self._r_hc_b[:] = compose_fixed_offset_pose(
            position_h_b, quaternion_h_b, self._offset_position_h, self._offset_quaternion_h
        )
        self._c_pose_b[:, :3] = position_c_b
        self._c_pose_b[:, 3:] = quaternion_c_b

        # body_link_vel_w is explicitly at H's actor/link origin, unlike the
        # legacy body_vel_w alias whose linear part is at the COM.
        root_to_h_w = body_position_w - root_position_w
        base_velocity_at_h_w = root_twist_w[:, :3] + torch.linalg.cross(
            root_twist_w[:, 3:], root_to_h_w, dim=-1
        )
        linear_h_b = quaternion_rotate(quaternion_bw, body_twist_w[:, :3] - base_velocity_at_h_w)
        angular_h_b = quaternion_rotate(quaternion_bw, body_twist_w[:, 3:] - root_twist_w[:, 3:])
        twist_h_b = torch.cat((linear_h_b, angular_h_b), dim=-1)
        self._c_twist_b[:] = shift_twist_to_point(twist_h_b, self._r_hc_b)

    def _compute_c_jacobian(self) -> None:
        jacobian_w_full = self._asset.root_physx_view.get_jacobians()[:, self._jacobian_body_idx]
        # PhysX linear Jacobian rows are evaluated at the link COM.  The
        # angular rows are point-independent, so rotate both blocks into B and
        # then shift from COM directly to C.  Shifting by H-to-C here would be
        # wrong whenever H has a non-zero inertial origin.
        jacobian_com_w = torch.index_select(jacobian_w_full, 2, self._joint_ids_tensor)
        quaternion_bw = quaternion_conjugate(_normalize_quaternion(self._asset.data.root_link_quat_w))
        rotation_bw = quaternion_to_matrix(quaternion_bw)
        jacobian_com_b = torch.empty_like(jacobian_com_w)
        jacobian_com_b[:, :3] = rotation_bw @ jacobian_com_w[:, :3]
        jacobian_com_b[:, 3:] = rotation_bw @ jacobian_com_w[:, 3:]

        body_quaternion_w = _normalize_quaternion(
            self._asset.data.body_link_quat_w[:, self._body_idx]
        )
        quaternion_h_b = _normalize_quaternion(
            quaternion_multiply(quaternion_bw, body_quaternion_w)
        )
        r_comc_b = com_to_target_offset(
            quaternion_h_b,
            self._asset.data.body_com_pos_b[:, self._body_idx],
            self._r_hc_b,
        )
        self._c_jacobian_b[:] = shift_jacobian_to_point(jacobian_com_b, r_comc_b)


@configclass
class CurrentFrameOscActionCfg(ActionTermCfg):
    """Configuration for :class:`CurrentFrameOscAction`."""

    class_type: type[ActionTerm] = CurrentFrameOscAction
    joint_names: list[str] = list(UR5E_ARM_JOINT_NAMES)
    body_name: str = "inspire_base_link"
    body_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_offset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    translation_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    rotation_scale: tuple[float, float, float] = (0.05, 0.05, 0.05)
    motion_stiffness: tuple[float, float, float, float, float, float] = (
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    )
    motion_damping_ratio: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    gravity_compensation: bool = True
    # Full operational-space inertia is required for a stable fixed-gain
    # acceleration law on the low-inertia UR5e wrist/hand assembly.
    inertial_dynamics_decoupling: bool = True
    partial_inertial_dynamics_decoupling: bool = False
    effort_limits: tuple[float, float, float, float, float, float] = UR5E_EFFORT_LIMITS_NM
    effort_limit_scale: float = 0.9
    saturation_tolerance: float = 1.0e-6


class InspireHandSynergyAction(ActionTerm):
    """Two-dimensional rate-limited Inspire position action.

    Policy inputs use the runner's conventional ``[-1, 1]`` range and are
    mapped linearly to the configured physical openness interval.  The default
    interval is ``[0, 1]``, for which ``u = (a + 1) / 2``.  Coordinate zero controls common
    flexion (thumb 2 plus four finger masters and all their followers), while
    coordinate one independently controls thumb 1.  The public
    :attr:`actual_synergy` is computed from actual master-joint positions, not
    copied from the command.
    """

    cfg: "InspireHandSynergyActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "InspireHandSynergyActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError(f"{self.__class__.__name__} requires an Articulation asset.")
        self._joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(joint_names) != tuple(cfg.joint_names) or tuple(joint_names) != INSPIRE_HAND_JOINT_NAMES:
            raise ValueError(
                "Hand joints must resolve exactly in the standalone Inspire order; "
                f"configured={cfg.joint_names}, resolved={joint_names}."
            )
        self._joint_ids_tensor = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._open_positions = _as_hand_vector(cfg.open_positions, "open_positions", self.device)
        self._closed_positions = _as_hand_vector(cfg.closed_positions, "closed_positions", self.device)
        if torch.any(torch.abs(self._open_positions - self._closed_positions) < 1.0e-9):
            raise ValueError("Every Inspire joint needs distinct open and closed positions.")

        if len(cfg.synergy_range) != 2:
            raise ValueError("synergy_range must contain the minimum and maximum physical openness.")
        self._synergy_low, self._synergy_high = (float(value) for value in cfg.synergy_range)
        if not (
            math.isfinite(self._synergy_low)
            and math.isfinite(self._synergy_high)
            and 0.0 <= self._synergy_low < self._synergy_high <= 1.0
        ):
            raise ValueError("synergy_range must satisfy 0 <= minimum < maximum <= 1.")
        self._synergy_span = self._synergy_high - self._synergy_low

        self._synergy_rate = torch.as_tensor(cfg.max_synergy_rate, dtype=torch.float32, device=self.device)
        if self._synergy_rate.shape != (2,) or not torch.isfinite(self._synergy_rate).all() or torch.any(
            self._synergy_rate <= 0.0
        ):
            raise ValueError("max_synergy_rate must contain two finite positive values.")
        self._joint_target_rate = _as_hand_vector(cfg.max_joint_target_rate, "max_joint_target_rate", self.device)
        if torch.any(self._joint_target_rate <= 0.0):
            raise ValueError("Every max_joint_target_rate must be positive.")
        if not math.isfinite(cfg.saturation_tolerance) or cfg.saturation_tolerance < 0.0:
            raise ValueError("saturation_tolerance must be finite and non-negative.")
        self._policy_dt = float(env.step_dt)
        if not torch.isfinite(torch.tensor(self._policy_dt)) or self._policy_dt <= 0.0:
            raise ValueError(f"Environment step_dt must be positive, got {self._policy_dt}.")

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._synergy_target = torch.full_like(self._raw_actions, 0.5)
        self._joint_targets = torch.zeros(
            self.num_envs, len(INSPIRE_HAND_JOINT_NAMES), dtype=torch.float32, device=self.device
        )
        self._action_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._synergy_joint_target_limits: torch.Tensor | None = None

        # Manager construction happens after the first simulator reset, so the
        # state buffers are valid here. Preserve actual targets when in range;
        # a restricted task must project an out-of-range startup state before
        # any target can be sent to the drives. Reset events then establish the
        # task's physical initial hand state before normal policy steps.
        self._sync_targets_to_current(slice(None))
        if cfg.enforce_synergy_joint_limits:
            self._install_synergy_joint_limits()
            self._sync_targets_to_current(slice(None))

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """Sanitized and clipped normalized policy actions in ``[-1, 1]``."""

        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Executed targets normalized to the configured range in ``[-1, 1]``."""

        return self._processed_actions

    @property
    def synergy_target(self) -> torch.Tensor:
        """Rate-limited target openness ``[common_flexion, thumb_1]`` in ``[0, 1]``."""

        return self._synergy_target

    @property
    def joint_targets(self) -> torch.Tensor:
        return self._joint_targets

    @property
    def actual_synergy(self) -> torch.Tensor:
        """Actual two-dimensional hand openness, aggregated without followers."""

        joint_positions = torch.index_select(self._asset.data.joint_pos, 1, self._joint_ids_tensor)
        return inspire_joint_positions_to_synergy(joint_positions, self._open_positions, self._closed_positions)

    def action_to_synergy(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Map normalized ``[-1, 1]`` actions to physical hand openness."""

        return self._synergy_low + 0.5 * (normalized_actions + 1.0) * self._synergy_span

    def synergy_to_action(self, synergy: torch.Tensor) -> torch.Tensor:
        """Normalize physical openness for hold actions and executed history.

        Values beyond the task's allowed interval map to its nearest endpoint.
        The physical state observation itself remains in the original ``[0, 1]``
        openness coordinates.
        """

        return (2.0 * (synergy - self._synergy_low) / self._synergy_span - 1.0).clamp(-1.0, 1.0)

    @property
    def action_saturated(self) -> torch.Tensor:
        return self._action_saturated

    def process_actions(self, actions: torch.Tensor) -> None:
        # Clear once per policy step.
        self._action_saturated[:] = False
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.process_actions_for_envs(actions, env_ids)

    def process_actions_for_envs(
        self, actions: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Process hand targets only for selected vector environments."""

        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if index.ndim != 1 or index.numel() == 0:
            if index.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            return
        if (index < 0).any() or (index >= self.num_envs).any():
            raise IndexError("env_ids contains an out-of-range environment index")
        expected_shape = (len(index), self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected hand actions with shape {expected_shape}, got {tuple(actions.shape)}.")
        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, torch.zeros_like(actions))
        normalized = sanitized.clamp(-1.0, 1.0)
        self._raw_actions[index] = normalized
        desired_synergy = self.action_to_synergy(normalized)

        max_synergy_delta = self._synergy_rate * self._policy_dt
        synergy_delta = torch.clamp(
            desired_synergy - self._synergy_target[index],
            min=-max_synergy_delta,
            max=max_synergy_delta,
        )
        next_synergy = (self._synergy_target[index] + synergy_delta).clamp(
            self._synergy_low, self._synergy_high
        )
        desired_joint_targets = inspire_synergy_to_joint_positions(
            next_synergy, self._open_positions, self._closed_positions
        )

        max_joint_delta = self._joint_target_rate * self._policy_dt
        joint_delta = torch.clamp(
            desired_joint_targets - self._joint_targets[index],
            min=-max_joint_delta,
            max=max_joint_delta,
        )
        next_joint_targets = self._joint_targets[index] + joint_delta
        if self._synergy_joint_target_limits is None:
            joint_limits = torch.index_select(
                self._asset.data.soft_joint_pos_limits[index], 1, self._joint_ids_tensor
            )
        else:
            joint_limits = self._synergy_joint_target_limits[index]
        next_joint_targets = torch.clamp(next_joint_targets, min=joint_limits[..., 0], max=joint_limits[..., 1])

        # Joint-rate limiting may keep the physical target behind the requested
        # synergy.  Re-aggregate the executed target so action-rate rewards use
        # what was actually sent to the drives.
        self._joint_targets[index] = next_joint_targets
        self._synergy_target[index] = inspire_joint_positions_to_synergy(
            self._joint_targets[index], self._open_positions, self._closed_positions
        )
        self._processed_actions[index] = self.synergy_to_action(self._synergy_target[index])

        input_clipped = torch.any(sanitized != normalized, dim=-1)
        synergy_limited = torch.any(torch.abs(desired_synergy - next_synergy) > self.cfg.saturation_tolerance, dim=-1)
        joint_limited = torch.any(
            torch.abs(desired_joint_targets - next_joint_targets) > self.cfg.saturation_tolerance, dim=-1
        )
        self._action_saturated[index] |= (
            (~finite.all(dim=-1)) | input_clipped | synergy_limited | joint_limited
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            index: slice | torch.Tensor = slice(None)
            sim_env_ids = None
        else:
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            sim_env_ids = index
        self._sync_targets_to_current(index)
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=sim_env_ids
        )

    def _sync_targets_to_current(self, index: slice | torch.Tensor) -> None:
        actual_joint_positions = torch.index_select(self._asset.data.joint_pos, 1, self._joint_ids_tensor)[index]
        actual_synergy = inspire_joint_positions_to_synergy(
            actual_joint_positions, self._open_positions, self._closed_positions
        )
        target_synergy = actual_synergy.clamp(self._synergy_low, self._synergy_high)
        projected_joint_positions = inspire_synergy_to_joint_positions(
            target_synergy, self._open_positions, self._closed_positions
        )
        outside_range = torch.any(actual_synergy != target_synergy, dim=-1, keepdim=True)
        joint_targets = torch.where(outside_range, projected_joint_positions, actual_joint_positions)
        if self._synergy_joint_target_limits is not None:
            joint_limits = self._synergy_joint_target_limits[index]
            joint_targets = joint_targets.clamp(min=joint_limits[..., 0], max=joint_limits[..., 1])
        self._joint_targets[index] = joint_targets
        self._synergy_target[index] = inspire_joint_positions_to_synergy(
            joint_targets, self._open_positions, self._closed_positions
        )
        self._raw_actions[index] = self.synergy_to_action(self._synergy_target[index])
        self._processed_actions[index] = self._raw_actions[index]
        self._action_saturated[index] = False

    def _install_synergy_joint_limits(self) -> None:
        """Install the physical hand window once, preserving original limits."""

        first_endpoint = self._closed_positions + self._synergy_low * (self._open_positions - self._closed_positions)
        second_endpoint = self._closed_positions + self._synergy_high * (self._open_positions - self._closed_positions)
        window_low = torch.minimum(first_endpoint, second_endpoint)
        window_high = torch.maximum(first_endpoint, second_endpoint)
        original_hard_limits = torch.index_select(self._asset.data.joint_pos_limits, 1, self._joint_ids_tensor)
        original_soft_limits = torch.index_select(self._asset.data.soft_joint_pos_limits, 1, self._joint_ids_tensor)
        physical_limits = torch.stack((
            torch.maximum(original_hard_limits[..., 0], window_low),
            torch.minimum(original_hard_limits[..., 1], window_high),
        ), dim=-1)
        target_limits = torch.stack((
            torch.maximum(original_soft_limits[..., 0], physical_limits[..., 0]),
            torch.minimum(original_soft_limits[..., 1], physical_limits[..., 1]),
        ), dim=-1)
        if torch.any(physical_limits[..., 0] > physical_limits[..., 1]):
            raise ValueError("synergy_range does not intersect the existing physical hand joint limits.")
        if torch.any(target_limits[..., 0] > target_limits[..., 1]):
            raise ValueError("synergy_range does not intersect the existing soft hand joint limits.")
        # Articulation recomputes soft limits about the new hard window. Keep
        # the intersection with the original soft limits so that the configured
        # open endpoint is not narrowed a second time by the soft-limit factor.
        self._synergy_joint_target_limits = target_limits
        self._asset.write_joint_position_limit_to_sim(
            physical_limits, joint_ids=self._joint_ids, warn_limit_violation=False
        )

    def reset_to_current(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Make position targets equal the hand state written by reset events."""

        self.reset(env_ids)


@configclass
class InspireHandSynergyActionCfg(ActionTermCfg):
    """Configuration for :class:`InspireHandSynergyAction`."""

    class_type: type[ActionTerm] = InspireHandSynergyAction
    joint_names: list[str] = list(INSPIRE_HAND_JOINT_NAMES)
    open_positions: tuple[float, ...] = INSPIRE_OPEN_JOINT_POSITIONS
    closed_positions: tuple[float, ...] = INSPIRE_CLOSED_JOINT_POSITIONS
    # Both coordinates are physical openness: 0 closed, 1 fully extended.
    # Tasks may restrict their allowed interval without changing action size.
    synergy_range: tuple[float, float] = (0.0, 1.0)
    # Optional physical constraint on all twelve joints. Installed only once;
    # finite solver tolerance may allow small measured deviations at a limit.
    enforce_synergy_joint_limits: bool = False
    max_synergy_rate: tuple[float, float] = (2.0, 2.0)
    max_joint_target_rate: tuple[float, ...] = (1.0,) * len(INSPIRE_HAND_JOINT_NAMES)
    saturation_tolerance: float = 1.0e-6


# Descriptive aliases used by configuration code and downstream probes.
ControlFrameOscAction = CurrentFrameOscAction
ControlFrameOscActionCfg = CurrentFrameOscActionCfg
InspireSynergyAction = InspireHandSynergyAction
InspireSynergyActionCfg = InspireHandSynergyActionCfg
