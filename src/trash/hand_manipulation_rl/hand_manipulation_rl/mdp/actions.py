"""Task-specific arm and Inspire hand actions for blind sweeping.

The arm action is deliberately not a thin wrapper around Isaac Lab's
``pose_rel`` action.  A normalized six-dimensional increment is interpreted in
the *current, measured* virtual control frame C, converted to one absolute
target per policy step, and held for every physics substep.  The pose, twist,
and geometric Jacobian supplied to OSC are all evaluated at the same virtual
point C.

The hand action expands two normalized policy values into position targets for
all twelve movable Inspire joints.  The follower ratios are copied from the
source Inspire URDF so this package does not depend on the asset-generation
package at run time.

Quaternion arguments use Isaac Lab's ``(w, x, y, z)`` convention throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Do not infer arm joints from articulation order: the assembled robot also has
# twelve hand degrees of freedom.
UR5E_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

UR5E_EFFORT_LIMITS_NM: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)

# Stable order from src/inspire_robot/urdf_left_with_force_sensor.  The USD
# conversion expands the mimic graph into normal driven joints, hence followers
# are commanded explicitly below.
INSPIRE_HAND_JOINT_NAMES: tuple[str, ...] = (
    "inspire_left_thumb_1_joint",
    "inspire_left_thumb_2_joint",
    "inspire_left_thumb_3_joint",
    "inspire_left_thumb_4_joint",
    "inspire_left_index_1_joint",
    "inspire_left_index_2_joint",
    "inspire_left_middle_1_joint",
    "inspire_left_middle_2_joint",
    "inspire_left_ring_1_joint",
    "inspire_left_ring_2_joint",
    "inspire_left_little_1_joint",
    "inspire_left_little_2_joint",
)

_THUMB_2_CLOSED = 0.5864
_THUMB_3_MIMIC_RATIO = 0.8024
_THUMB_4_MIMIC_RATIO = 0.9487
_FINGER_MASTER_CLOSED = 1.4381
_FINGER_FOLLOWER_RATIO = 1.0843

INSPIRE_OPEN_JOINT_POSITIONS: tuple[float, ...] = (
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

INSPIRE_CLOSED_JOINT_POSITIONS: tuple[float, ...] = (
    1.1641,
    _THUMB_2_CLOSED,
    _THUMB_2_CLOSED * _THUMB_3_MIMIC_RATIO,
    _THUMB_2_CLOSED * _THUMB_3_MIMIC_RATIO * _THUMB_4_MIMIC_RATIO,
    _FINGER_MASTER_CLOSED,
    _FINGER_MASTER_CLOSED * _FINGER_FOLLOWER_RATIO,
    _FINGER_MASTER_CLOSED,
    _FINGER_MASTER_CLOSED * _FINGER_FOLLOWER_RATIO,
    _FINGER_MASTER_CLOSED,
    _FINGER_MASTER_CLOSED * _FINGER_FOLLOWER_RATIO,
    _FINGER_MASTER_CLOSED,
    _FINGER_MASTER_CLOSED * _FINGER_FOLLOWER_RATIO,
)

# Common flexion is estimated only from independently driven source joints;
# follower joints must not be counted a second time.
INSPIRE_FLEXION_MASTER_INDICES: tuple[int, ...] = (1, 4, 6, 8, 10)
INSPIRE_THUMB_1_INDEX = 0


def _normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Normalize batched wxyz quaternions without a host/device synchronization."""

    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return quaternion / torch.clamp_min(norm, eps)


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of batched wxyz quaternions."""

    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product of broadcast-compatible wxyz quaternions."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit wxyz quaternions."""

    quaternion = _normalize_quaternion(quaternion)
    q_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(q_vector, twice_cross, dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert batched unit wxyz quaternions to rotation matrices."""

    quaternion = _normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_from_rotation_vector(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation vectors to unit wxyz quaternions.

    A series expansion keeps both the value and gradient finite around zero.
    """

    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    small = angle < 1.0e-6
    scale = torch.where(
        small,
        0.5 - angle.square() / 48.0 + angle.pow(4) / 3840.0,
        torch.sin(half_angle) / torch.clamp_min(angle, 1.0e-12),
    )
    quaternion = torch.cat((torch.cos(half_angle), rotation_vector * scale), dim=-1)
    return _normalize_quaternion(quaternion)


def compose_fixed_offset_pose(
    parent_position_b: torch.Tensor,
    parent_quaternion_b: torch.Tensor,
    offset_position_h: torch.Tensor,
    offset_quaternion_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``T_BC = T_BH T_HC`` and return ``r_HC`` expressed in B."""

    r_hc_b = quaternion_rotate(parent_quaternion_b, offset_position_h)
    position_c_b = parent_position_b + r_hc_b
    quaternion_c_b = quaternion_multiply(parent_quaternion_b, offset_quaternion_h)
    return position_c_b, _normalize_quaternion(quaternion_c_b), r_hc_b


def current_frame_delta_target(
    current_position_b: torch.Tensor,
    current_quaternion_b: torch.Tensor,
    delta_position_c: torch.Tensor,
    delta_rotation_vector_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Apply a right/local increment to the current actual C pose.

    This implements ``p_des = p_C + R_BC dp_C`` and
    ``R_des = R_BC Exp(dphi_C)``.  It is intentionally different from a
    stationary-frame ``pose_rel`` update.
    """

    target_position_b = current_position_b + quaternion_rotate(current_quaternion_b, delta_position_c)
    delta_quaternion_c = quaternion_from_rotation_vector(delta_rotation_vector_c)
    target_quaternion_b = quaternion_multiply(current_quaternion_b, delta_quaternion_c)
    return target_position_b, _normalize_quaternion(target_quaternion_b)


def task_frame_translation_delta_w(
    normalized_translation: torch.Tensor,
    command_direction_w: torch.Tensor,
    translation_scale: torch.Tensor | tuple[float, float, float],
) -> torch.Tensor:
    """Map policy translation axes to stable task/world directions.

    The policy axes are ``[world-up, toward-object, shelf-depth(+X)]``.  In
    particular, a positive second coordinate always moves toward the object;
    it does not change sign between palm/dorsal modes or left/right commands.
    """

    if normalized_translation.ndim != 2 or normalized_translation.shape[-1] != 3:
        raise ValueError("normalized_translation must have shape (N, 3)")
    if command_direction_w.shape != normalized_translation.shape:
        raise ValueError("command_direction_w must have shape (N, 3)")
    scale = torch.as_tensor(
        translation_scale,
        dtype=normalized_translation.dtype,
        device=normalized_translation.device,
    )
    if scale.shape != (3,) or not torch.isfinite(scale).all() or torch.any(scale <= 0.0):
        raise ValueError("translation_scale must contain three finite positive values")
    direction_norm = torch.linalg.vector_norm(command_direction_w, dim=-1, keepdim=True)
    if torch.any(direction_norm <= 1.0e-8):
        raise ValueError("command_direction_w must be non-zero")
    return _task_frame_translation_delta_w_impl(
        normalized_translation,
        command_direction_w,
        scale,
    )


def _task_frame_translation_delta_w_impl(
    normalized_translation: torch.Tensor,
    command_direction_w: torch.Tensor,
    translation_scale: torch.Tensor,
) -> torch.Tensor:
    """Hot-path implementation after episode/config validation."""

    direction_norm = torch.linalg.vector_norm(command_direction_w, dim=-1, keepdim=True)
    direction = command_direction_w / direction_norm

    delta_w = normalized_translation[:, 1:2] * translation_scale[1] * direction
    delta_w[:, 2] += normalized_translation[:, 0] * translation_scale[0]
    delta_w[:, 0] += normalized_translation[:, 2] * translation_scale[2]
    return delta_w


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Return batched cross-product matrices such that ``skew(v) @ x = v x x``."""

    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(vector.shape[:-1] + (3, 3))


def shift_jacobian_to_point(
    jacobian_source_b: torch.Tensor, r_source_target_b: torch.Tensor
) -> torch.Tensor:
    r"""Shift a geometric Jacobian between two points, all expressed in B.

    PhysX reports an articulation link's linear Jacobian at its center of
    mass, not at the link/actor origin.  Keeping the source point generic here
    makes that distinction explicit at the call site.
    """

    if jacobian_source_b.shape[-2] != 6 or r_source_target_b.shape[-1] != 3:
        raise ValueError("Expected a (..., 6, dof) Jacobian and (..., 3) offset.")
    jacobian_target_b = jacobian_source_b.clone()
    jacobian_target_b[..., :3, :] = (
        jacobian_source_b[..., :3, :]
        - skew_symmetric(r_source_target_b) @ jacobian_source_b[..., 3:, :]
    )
    return jacobian_target_b


def com_to_target_offset(
    parent_quaternion_b: torch.Tensor,
    parent_to_com_h: torch.Tensor,
    parent_to_target_b: torch.Tensor,
) -> torch.Tensor:
    r"""Return the COM-to-target vector in B for a target fixed to link H."""

    parent_to_com_b = quaternion_rotate(parent_quaternion_b, parent_to_com_h)
    return parent_to_target_b - parent_to_com_b


def shift_twist_to_point(twist_h_b: torch.Tensor, r_hc_b: torch.Tensor) -> torch.Tensor:
    r"""Shift ``[v_H, omega_H]`` to ``[v_C, omega_C]``, expressed in B."""

    if twist_h_b.shape[-1] != 6 or r_hc_b.shape[-1] != 3:
        raise ValueError("Expected a (..., 6) twist and (..., 3) offset.")
    linear_c_b = twist_h_b[..., :3] + torch.linalg.cross(twist_h_b[..., 3:], r_hc_b, dim=-1)
    return torch.cat((linear_c_b, twist_h_b[..., 3:]), dim=-1)


def inspire_synergy_to_joint_positions(
    synergy_open: torch.Tensor,
    open_positions: torch.Tensor | None = None,
    closed_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand ``[common_flexion_open, thumb_1_open]`` to twelve joints.

    Both synergy coordinates use ``0 = fully closed`` and ``1 = fully open``.
    """

    if synergy_open.shape[-1] != 2:
        raise ValueError(f"Expected two hand synergy values, got shape {tuple(synergy_open.shape)}.")
    if open_positions is None:
        open_positions = torch.as_tensor(
            INSPIRE_OPEN_JOINT_POSITIONS, dtype=synergy_open.dtype, device=synergy_open.device
        )
    if closed_positions is None:
        closed_positions = torch.as_tensor(
            INSPIRE_CLOSED_JOINT_POSITIONS, dtype=synergy_open.dtype, device=synergy_open.device
        )
    if open_positions.shape[-1] != len(INSPIRE_HAND_JOINT_NAMES) or closed_positions.shape[-1] != len(
        INSPIRE_HAND_JOINT_NAMES
    ):
        raise ValueError("Open and closed maps must contain twelve Inspire joint positions.")

    openness = synergy_open[..., :1].expand(synergy_open.shape[:-1] + (len(INSPIRE_HAND_JOINT_NAMES),)).clone()
    openness[..., INSPIRE_THUMB_1_INDEX] = synergy_open[..., 1]
    return closed_positions + openness * (open_positions - closed_positions)


def inspire_joint_positions_to_synergy(
    joint_positions: torch.Tensor,
    open_positions: torch.Tensor | None = None,
    closed_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate actual master-joint state into two openness coordinates."""

    if joint_positions.shape[-1] != len(INSPIRE_HAND_JOINT_NAMES):
        raise ValueError(f"Expected twelve Inspire joint positions, got shape {tuple(joint_positions.shape)}.")
    if open_positions is None:
        open_positions = torch.as_tensor(
            INSPIRE_OPEN_JOINT_POSITIONS, dtype=joint_positions.dtype, device=joint_positions.device
        )
    if closed_positions is None:
        closed_positions = torch.as_tensor(
            INSPIRE_CLOSED_JOINT_POSITIONS, dtype=joint_positions.dtype, device=joint_positions.device
        )
    denominator = open_positions - closed_positions
    joint_openness = ((joint_positions - closed_positions) / denominator).clamp(0.0, 1.0)
    master_indices = torch.as_tensor(
        INSPIRE_FLEXION_MASTER_INDICES, dtype=torch.long, device=joint_positions.device
    )
    common_open = torch.index_select(joint_openness, dim=-1, index=master_indices).mean(dim=-1)
    thumb_1_open = joint_openness[..., INSPIRE_THUMB_1_INDEX]
    return torch.stack((common_open, thumb_1_open), dim=-1)


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
            raise ValueError("Blind-sweep OSC requires the configured fixed-base UR5e articulation.")

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
        # Start one policy-step diagnostic window. Partial reset-validation
        # releases below must accumulate into, rather than erase, flags raised
        # by the initial hold substep.
        self._invalid_action[:] = False
        self._torque_saturated[:] = False
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.process_actions_for_envs(actions, env_ids)

    def process_actions_for_envs(
        self, actions: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Process new commands only for selected vector environments.

        The environment uses this after the first physics substep of a fresh
        reset: contact tensors validate the untouched reset pose first, then
        the caller's policy command is released for the remaining substeps.
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
        computed_efforts += self._additional_joint_efforts()
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

    def _additional_joint_efforts(self) -> torch.Tensor:
        """Return optional task-specific feed-forward efforts."""

        return torch.zeros_like(self._joint_efforts)

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


class TaskFrameOscAction(CurrentFrameOscAction):
    """OSC action with mode-invariant task-aligned translation axes.

    Translation inputs are interpreted as ``[world-up, toward-object,
    shelf-depth(+X)]`` and then expressed in the measured current C frame for
    the inherited relative-pose target.  Rotation inputs retain the parent's
    current-C-frame convention.
    """

    cfg: "TaskFrameOscActionCfg"

    def __init__(self, cfg: "TaskFrameOscActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        if not math.isfinite(cfg.contact_push_force_n) or cfg.contact_push_force_n < 0.0:
            raise ValueError("contact_push_force_n must be finite and non-negative")
        if (
            not math.isfinite(cfg.contact_push_force_guard_n)
            or cfg.contact_push_force_guard_n <= 0.0
        ):
            raise ValueError("contact_push_force_guard_n must be finite and positive")
        if (
            not math.isfinite(cfg.contact_push_board_guard_n)
            or cfg.contact_push_board_guard_n <= 0.0
        ):
            raise ValueError("contact_push_board_guard_n must be finite and positive")
        self._task_env = env
        self._task_translation_scale = self._translation_scale.clone()
        self._contact_push_force_n = float(cfg.contact_push_force_n)
        self._contact_push_force_guard_n = float(cfg.contact_push_force_guard_n)
        self._contact_push_board_guard_n = float(cfg.contact_push_board_guard_n)

    def process_actions_for_envs(
        self, actions: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if index.ndim != 1 or index.numel() == 0:
            if index.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            return
        if (index < 0).any() or (index >= self.num_envs).any():
            raise IndexError("env_ids contains an out-of-range environment index")
        expected_shape = (len(index), self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected task-frame OSC actions with shape {expected_shape}, got {tuple(actions.shape)}."
            )

        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, torch.zeros_like(actions))
        normalized = sanitized.clamp(-1.0, 1.0)
        self._raw_actions[index] = normalized
        invalid_action = ~finite.all(dim=-1)
        self._invalid_action[index] |= invalid_action
        self._torque_saturated[index] |= invalid_action | torch.any(
            sanitized != normalized, dim=-1
        )

        delta_w = _task_frame_translation_delta_w_impl(
            normalized[:, :3],
            self._task_env.command_direction_w[index],
            self._task_translation_scale,
        )
        root_quaternion_w = _normalize_quaternion(
            self._asset.data.root_link_quat_w[index]
        )
        delta_b = quaternion_rotate(quaternion_conjugate(root_quaternion_w), delta_w)
        self._compute_c_pose_and_twist()
        delta_c = quaternion_rotate(
            quaternion_conjugate(self._c_pose_b[index, 3:]), delta_b
        )
        self._processed_actions[index, :3] = delta_c
        self._processed_actions[index, 3:] = normalized[:, 3:] * self._rotation_scale

        target_position_b, target_quaternion_b = current_frame_delta_target(
            self._c_pose_b[index, :3],
            self._c_pose_b[index, 3:],
            self._processed_actions[index, :3],
            self._processed_actions[index, 3:],
        )
        self._desired_c_pose_b[index, :3] = target_position_b
        self._desired_c_pose_b[index, 3:] = target_quaternion_b
        self._osc.set_command(self._desired_c_pose_b)

    def _additional_joint_efforts(self) -> torch.Tensor:
        """Apply a bounded forward preload only during live Hand--Cube contact.

        The measured-pose relative OSC deliberately keeps free-space increments
        small.  At contact that also bounds the pose error, and therefore the
        transmitted force, below the Cube's breakaway friction.  This explicit
        task-space preload preserves the safe approach dynamics while making a
        continued positive approach action capable of initiating a push.
        """

        if self._contact_push_force_n == 0.0:
            return torch.zeros_like(self._joint_efforts)
        sensor_live = self._task_env.sensor_valid & self._task_env.sensor_data_fresh
        contact = self._task_env.hand_object_contact() & sensor_live
        below_force_guard = (
            self._task_env.hand_object_contact_force_n()
            < self._contact_push_force_guard_n
        )
        below_board_guard = (
            self._task_env.board_force < self._contact_push_board_guard_n
        )
        contact &= below_force_guard & below_board_guard
        forward_command = torch.clamp(self._raw_actions[:, 1], min=0.0, max=1.0)
        force_magnitude = (
            contact.to(dtype=torch.float32)
            * forward_command
            * self._contact_push_force_n
        )
        force_w = self._task_env.command_direction_w * force_magnitude.unsqueeze(-1)
        root_quaternion_w = _normalize_quaternion(self._asset.data.root_link_quat_w)
        force_b = quaternion_rotate(quaternion_conjugate(root_quaternion_w), force_w)
        return torch.bmm(
            self._c_jacobian_b[:, :3].mT,
            force_b.unsqueeze(-1),
        ).squeeze(-1)


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
    gravity_compensation: bool = False
    # Used only by TaskFrameOscAction. Zero preserves the base/v0 controller.
    contact_push_force_n: float = 0.0
    contact_push_force_guard_n: float = 12.0
    contact_push_board_guard_n: float = 2.0
    # Full operational-space inertia is required for a stable fixed-gain
    # acceleration law on the low-inertia UR5e wrist/hand assembly.
    inertial_dynamics_decoupling: bool = True
    partial_inertial_dynamics_decoupling: bool = False
    effort_limits: tuple[float, float, float, float, float, float] = UR5E_EFFORT_LIMITS_NM
    effort_limit_scale: float = 0.9
    saturation_tolerance: float = 1.0e-6


@configclass
class TaskFrameOscActionCfg(CurrentFrameOscActionCfg):
    """Configuration for :class:`TaskFrameOscAction`.

    ``translation_scale`` is ordered as world-up, toward-object, and world-X
    shelf-depth rather than as current-C local XYZ.
    """

    class_type: type[ActionTerm] = TaskFrameOscAction


class InspireHandSynergyAction(ActionTerm):
    """Two-dimensional rate-limited Inspire position action.

    Policy inputs use the runner's conventional ``[-1, 1]`` range and are
    mapped to openness ``u = (a + 1) / 2``.  Coordinate zero controls common
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

        # Manager construction happens after the first simulator reset, so the
        # state buffers are valid here.  Start from actual positions to avoid a
        # target jump before ActionManager.reset() is called.
        self._joint_targets[:] = torch.index_select(self._asset.data.joint_pos, 1, self._joint_ids_tensor)
        self._synergy_target[:] = inspire_joint_positions_to_synergy(
            self._joint_targets, self._open_positions, self._closed_positions
        )
        self._processed_actions[:] = 2.0 * self._synergy_target - 1.0

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """Sanitized and clipped normalized policy actions in ``[-1, 1]``."""

        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Actually executed, rate-limited normalized actions in ``[-1, 1]``."""

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

    @property
    def action_saturated(self) -> torch.Tensor:
        return self._action_saturated

    def process_actions(self, actions: torch.Tensor) -> None:
        # Clear once per policy step; a later partial release ORs its own
        # rate/clipping status into this same diagnostic window.
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
        desired_synergy = 0.5 * (normalized + 1.0)

        max_synergy_delta = self._synergy_rate * self._policy_dt
        synergy_delta = torch.clamp(
            desired_synergy - self._synergy_target[index],
            min=-max_synergy_delta,
            max=max_synergy_delta,
        )
        next_synergy = (self._synergy_target[index] + synergy_delta).clamp(0.0, 1.0)
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
        joint_limits = torch.index_select(
            self._asset.data.soft_joint_pos_limits[index], 1, self._joint_ids_tensor
        )
        next_joint_targets = torch.clamp(next_joint_targets, min=joint_limits[..., 0], max=joint_limits[..., 1])

        # Joint-rate limiting may keep the physical target behind the requested
        # synergy.  Re-aggregate the executed target so action-rate rewards use
        # what was actually sent to the drives.
        self._joint_targets[index] = next_joint_targets
        self._synergy_target[index] = inspire_joint_positions_to_synergy(
            self._joint_targets[index], self._open_positions, self._closed_positions
        )
        self._processed_actions[index] = 2.0 * self._synergy_target[index] - 1.0

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
        actual_joint_positions = torch.index_select(self._asset.data.joint_pos, 1, self._joint_ids_tensor)
        self._joint_targets[index] = actual_joint_positions[index]
        actual_synergy = inspire_joint_positions_to_synergy(
            actual_joint_positions[index], self._open_positions, self._closed_positions
        )
        self._synergy_target[index] = actual_synergy
        self._raw_actions[index] = 2.0 * actual_synergy - 1.0
        self._processed_actions[index] = self._raw_actions[index]
        self._action_saturated[index] = False
        self._asset.set_joint_position_target(
            self._joint_targets[index], joint_ids=self._joint_ids, env_ids=sim_env_ids
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
    max_synergy_rate: tuple[float, float] = (2.0, 2.0)
    max_joint_target_rate: tuple[float, ...] = (1.0,) * len(INSPIRE_HAND_JOINT_NAMES)
    saturation_tolerance: float = 1.0e-6


# Descriptive aliases used by configuration code and downstream probes.
ControlFrameOscAction = CurrentFrameOscAction
ControlFrameOscActionCfg = CurrentFrameOscActionCfg
InspireSynergyAction = InspireHandSynergyAction
InspireSynergyActionCfg = InspireHandSynergyActionCfg
