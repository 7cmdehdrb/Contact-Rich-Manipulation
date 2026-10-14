"""Simulator-independent quaternion, OSC frame, and Inspire synergy math."""

from __future__ import annotations

import torch

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


