"""Sensor adapters for the blind-sweeping task.

This module deliberately depends only on :mod:`torch`.  Isaac Lab assets and
sensors are accepted through their public, duck-typed attributes (for example
``sensor.body_names`` and ``sensor.data.net_forces_w``).  Keeping the numerical
contract here makes it possible to unit-test channel ordering, wrench frames,
and force aggregation without starting Isaac Sim.

The supplied Inspire model has physical collision bodies only for the 17
palmar pads.  The dorsal channels below are therefore an explicit *parent-body
contact approximation*: each logical dorsal channel reads the net contact
force on the moving hand link that carries the corresponding palmar pad.  Two
logical regions on the same distal link consequently receive the same signal.
This is useful as a conservative bring-up signal, but it is not a taxel-level
or surface-side classifier.  In particular, it may respond to palmar or side
contacts on that parent link.  A future dorsal-region implementation can keep
the public 17-channel order while replacing only the force source.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


# Stable channel order inherited from the source Inspire hand.  Never use the
# incidental PhysX rigid-body order as an observation contract.
PALM_CHANNEL_NAMES: tuple[str, ...] = (
    "inspire_palm_force_sensor",
    "inspire_thumb_force_sensor_1",
    "inspire_thumb_force_sensor_2",
    "inspire_thumb_force_sensor_3",
    "inspire_thumb_force_sensor_4",
    "inspire_index_force_sensor_1",
    "inspire_index_force_sensor_2",
    "inspire_index_force_sensor_3",
    "inspire_middle_force_sensor_1",
    "inspire_middle_force_sensor_2",
    "inspire_middle_force_sensor_3",
    "inspire_ring_force_sensor_1",
    "inspire_ring_force_sensor_2",
    "inspire_ring_force_sensor_3",
    "inspire_little_force_sensor_1",
    "inspire_little_force_sensor_2",
    "inspire_little_force_sensor_3",
)

# Logical diagnostic labels.  These are not USD/URDF rigid-body names.
DORSAL_CHANNEL_NAMES: tuple[str, ...] = (
    "inspire_dorsal_palm_force_sensor",
    "inspire_dorsal_thumb_force_sensor_1",
    "inspire_dorsal_thumb_force_sensor_2",
    "inspire_dorsal_thumb_force_sensor_3",
    "inspire_dorsal_thumb_force_sensor_4",
    "inspire_dorsal_index_force_sensor_1",
    "inspire_dorsal_index_force_sensor_2",
    "inspire_dorsal_index_force_sensor_3",
    "inspire_dorsal_middle_force_sensor_1",
    "inspire_dorsal_middle_force_sensor_2",
    "inspire_dorsal_middle_force_sensor_3",
    "inspire_dorsal_ring_force_sensor_1",
    "inspire_dorsal_ring_force_sensor_2",
    "inspire_dorsal_ring_force_sensor_3",
    "inspire_dorsal_little_force_sensor_1",
    "inspire_dorsal_little_force_sensor_2",
    "inspire_dorsal_little_force_sensor_3",
)

# One entry per logical dorsal channel, in DORSAL_CHANNEL_NAMES order.  Repeated
# parent names are intentional: the physical model has 12 relevant parent
# bodies but the observation contract has 17 regions.
DORSAL_PARENT_BY_CHANNEL: tuple[str, ...] = (
    "inspire_base_link",
    "inspire_left_thumb_2",
    "inspire_left_thumb_3",
    "inspire_left_thumb_4",
    "inspire_left_thumb_4",
    "inspire_left_index_1",
    "inspire_left_index_2",
    "inspire_left_index_2",
    "inspire_left_middle_1",
    "inspire_left_middle_2",
    "inspire_left_middle_2",
    "inspire_left_ring_1",
    "inspire_left_ring_2",
    "inspire_left_ring_2",
    "inspire_left_little_1",
    "inspire_left_little_2",
    "inspire_left_little_2",
)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


DORSAL_PARENT_BODY_NAMES: tuple[str, ...] = _ordered_unique(DORSAL_PARENT_BY_CHANNEL)
DORSAL_PARENT_MAP: dict[str, str] = dict(zip(DORSAL_CHANNEL_NAMES, DORSAL_PARENT_BY_CHANNEL))

# Compatibility aliases with intentionally explicit semantics.
PALM_SENSOR_NAMES = PALM_CHANNEL_NAMES
DORSAL_SENSOR_NAMES = DORSAL_CHANNEL_NAMES
TACTILE_CHANNEL_COUNT_PER_SIDE = 17


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _name_indices(
    available_names: Sequence[str],
    requested_names: Sequence[str],
    *,
    require_exact_set: bool,
    source: str,
) -> tuple[int, ...]:
    available = tuple(available_names)
    if len(set(available)) != len(available):
        raise RuntimeError(f"{source} contains duplicate body names: {available}")
    missing = [name for name in requested_names if name not in available]
    if missing:
        raise RuntimeError(f"{source} is missing required bodies {missing}; available={available}")
    if require_exact_set and (len(available) != len(requested_names) or set(available) != set(requested_names)):
        extra = [name for name in available if name not in requested_names]
        raise RuntimeError(
            f"{source} must contain exactly the configured channel bodies; "
            f"extra={extra}, available={available}"
        )
    return tuple(available.index(name) for name in requested_names)


def _finite_force_vectors(forces_w: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(forces_w, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if forces_w.ndim < 2 or forces_w.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., channels, 3), got {tuple(forces_w.shape)}")
    if not torch.isfinite(forces_w).all():
        raise RuntimeError(f"{name} contains a non-finite contact force")
    return forces_w


def contact_magnitudes_and_bits(
    forces_w: torch.Tensor, threshold_n: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert world-frame normal-force vectors to magnitudes and binary bits.

    Thresholding is inclusive: a magnitude exactly equal to ``threshold_n`` is
    contact.  Isaac Lab's ``ContactSensorCfg.force_threshold`` controls contact
    time accounting and does not replace this observation threshold.
    """

    forces_w = _finite_force_vectors(forces_w, "forces_w")
    threshold_n = _positive_finite(threshold_n, "threshold_n")
    magnitudes_n = torch.linalg.vector_norm(forces_w, ord=2, dim=-1)
    bits = (magnitudes_n >= threshold_n).to(torch.uint8)
    return magnitudes_n, bits


@dataclass(frozen=True)
class BilateralTactileSample:
    """Batched bilateral tactile data in the stable palm-then-dorsal order."""

    palm_forces_w: torch.Tensor
    dorsal_forces_w: torch.Tensor
    forces_w: torch.Tensor
    palm_magnitudes_n: torch.Tensor
    dorsal_magnitudes_n: torch.Tensor
    magnitudes_n: torch.Tensor
    palm_bits: torch.Tensor
    dorsal_bits: torch.Tensor
    bits: torch.Tensor


class BilateralTactileReader:
    """Read 17 physical palm pads and 17 parent-body dorsal approximations.

    Args:
        palm_sensor: Contact sensor whose ``body_names`` are exactly the 17
            physical pad names and whose ``data.net_forces_w`` has shape
            ``(num_envs, 17, 3)``.
        dorsal_parent_sensor: Contact sensor containing the 12 parent bodies in
            :data:`DORSAL_PARENT_BODY_NAMES`.  Extra bodies are allowed, which
            permits a single robot-wide diagnostic ContactSensor.
        threshold_n: Configurable palm binary threshold in newtons.
        dorsal_threshold_n: Optional independent dorsal threshold.  It defaults
            to ``threshold_n``.

    ``net_forces_w`` is a net normal-contact force, not a friction-inclusive
    contact wrench.  The dorsal source does not distinguish which side of a
    parent link was touched and repeats a parent signal for logical regions
    sharing that link.  Consumers must not treat the 17 dorsal bits as 17
    independent force measurements.
    """

    def __init__(
        self,
        palm_sensor: Any,
        dorsal_parent_sensor: Any,
        *,
        threshold_n: float = 0.05,
        dorsal_threshold_n: float | None = None,
        palm_channel_names: Sequence[str] = PALM_CHANNEL_NAMES,
        dorsal_parent_by_channel: Sequence[str] = DORSAL_PARENT_BY_CHANNEL,
    ) -> None:
        if len(palm_channel_names) != TACTILE_CHANNEL_COUNT_PER_SIDE:
            raise ValueError("palm_channel_names must define exactly 17 channels")
        if len(dorsal_parent_by_channel) != TACTILE_CHANNEL_COUNT_PER_SIDE:
            raise ValueError("dorsal_parent_by_channel must define exactly 17 channels")
        self.palm_sensor = palm_sensor
        self.dorsal_parent_sensor = dorsal_parent_sensor
        self.palm_channel_names = tuple(palm_channel_names)
        self.dorsal_parent_by_channel = tuple(dorsal_parent_by_channel)
        self.threshold_n = _positive_finite(threshold_n, "threshold_n")
        self.dorsal_threshold_n = _positive_finite(
            self.threshold_n if dorsal_threshold_n is None else dorsal_threshold_n,
            "dorsal_threshold_n",
        )
        self._palm_indices = _name_indices(
            palm_sensor.body_names,
            self.palm_channel_names,
            require_exact_set=True,
            source="palm ContactSensor",
        )
        self._dorsal_indices = _name_indices(
            dorsal_parent_sensor.body_names,
            self.dorsal_parent_by_channel,
            require_exact_set=False,
            source="dorsal-parent ContactSensor",
        )

    @staticmethod
    def _select(forces_w: torch.Tensor, indices: Sequence[int], source: str) -> torch.Tensor:
        forces_w = _finite_force_vectors(forces_w, source)
        if forces_w.ndim != 3:
            raise ValueError(f"{source} must have shape (num_envs, bodies, 3), got {tuple(forces_w.shape)}")
        index = torch.as_tensor(indices, dtype=torch.long, device=forces_w.device)
        return torch.index_select(forces_w, dim=1, index=index)

    def read(self) -> BilateralTactileSample:
        palm_forces_w = self._select(
            self.palm_sensor.data.net_forces_w, self._palm_indices, "palm net_forces_w"
        )
        dorsal_forces_w = self._select(
            self.dorsal_parent_sensor.data.net_forces_w,
            self._dorsal_indices,
            "dorsal-parent net_forces_w",
        )
        if palm_forces_w.shape[0] != dorsal_forces_w.shape[0]:
            raise RuntimeError(
                "Palm and dorsal ContactSensors have different environment counts: "
                f"{palm_forces_w.shape[0]} != {dorsal_forces_w.shape[0]}"
            )
        palm_magnitudes_n, palm_bits = contact_magnitudes_and_bits(
            palm_forces_w, self.threshold_n
        )
        dorsal_magnitudes_n, dorsal_bits = contact_magnitudes_and_bits(
            dorsal_forces_w, self.dorsal_threshold_n
        )
        return BilateralTactileSample(
            palm_forces_w=palm_forces_w,
            dorsal_forces_w=dorsal_forces_w,
            forces_w=torch.cat((palm_forces_w, dorsal_forces_w), dim=1),
            palm_magnitudes_n=palm_magnitudes_n,
            dorsal_magnitudes_n=dorsal_magnitudes_n,
            magnitudes_n=torch.cat((palm_magnitudes_n, dorsal_magnitudes_n), dim=1),
            palm_bits=palm_bits,
            dorsal_bits=dorsal_bits,
            bits=torch.cat((palm_bits, dorsal_bits), dim=1),
        )


def transform_wrench_f_to_c(
    wrench_f: torch.Tensor,
    rotation_c_from_f: torch.Tensor,
    r_c_to_f_c: torch.Tensor,
) -> torch.Tensor:
    """Transform a wrench at sensor frame ``F`` to control point ``C``.

    ``rotation_c_from_f`` rotates vectors expressed in F into C.  The vector
    ``r_c_to_f_c`` points from C's origin to F's origin and is expressed in C.
    Therefore ``M_C = R_CF M_F + r_CF x (R_CF F_F)``.  Quaternion conversion,
    if needed, belongs at the caller boundary so this core has one unambiguous
    rotation convention.
    """

    if not isinstance(wrench_f, torch.Tensor) or wrench_f.ndim < 1 or wrench_f.shape[-1] != 6:
        raise ValueError(f"wrench_f must have shape (..., 6), got {getattr(wrench_f, 'shape', None)}")
    if not isinstance(rotation_c_from_f, torch.Tensor) or rotation_c_from_f.shape[-2:] != (3, 3):
        raise ValueError("rotation_c_from_f must have shape (..., 3, 3)")
    if not isinstance(r_c_to_f_c, torch.Tensor) or r_c_to_f_c.shape[-1:] != (3,):
        raise ValueError("r_c_to_f_c must have shape (..., 3)")
    if not (
        torch.isfinite(wrench_f).all()
        and torch.isfinite(rotation_c_from_f).all()
        and torch.isfinite(r_c_to_f_c).all()
    ):
        raise RuntimeError("Wrench transform inputs must be finite")

    force_f, moment_f = wrench_f[..., :3], wrench_f[..., 3:]
    force_c = torch.matmul(rotation_c_from_f, force_f.unsqueeze(-1)).squeeze(-1)
    moment_c_at_f = torch.matmul(rotation_c_from_f, moment_f.unsqueeze(-1)).squeeze(-1)
    try:
        moment_arm = torch.broadcast_to(r_c_to_f_c, force_c.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"r_c_to_f_c shape {tuple(r_c_to_f_c.shape)} cannot broadcast to {tuple(force_c.shape)}"
        ) from exc
    moment_c = moment_c_at_f + torch.cross(moment_arm, force_c, dim=-1)
    return torch.cat((force_c, moment_c), dim=-1)


@dataclass(frozen=True)
class FixedJointWrenchSample:
    """Uncancelled fixed-joint reaction in measurement and control frames.

    ``raw_f`` and ``measured_f`` intentionally contain the same incoming
    child-body reaction.  No gravity model, tare, or residual bias is removed,
    so Hand self-weight force and moment remain present.  ``measured_c`` only
    rotates the wrench and shifts its moment reference from F to C.
    """

    raw_f: torch.Tensor
    measured_f: torch.Tensor
    measured_c: torch.Tensor


def _exact_body_index(asset: Any, body_name: str) -> int:
    names = tuple(getattr(asset, "body_names", ()))
    if names:
        matches = [index for index, name in enumerate(names) if name == body_name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one body named {body_name!r}; body_names={names}")
        return matches[0]

    # Fallback for small adapters exposing only Isaac Lab's find_bodies API.
    if not hasattr(asset, "find_bodies"):
        raise AttributeError("Robot must expose body_names or find_bodies()")
    indices, matched_names = asset.find_bodies(body_name, preserve_order=True)
    indices = list(indices)
    matched_names = list(matched_names)
    exact = [(index, name) for index, name in zip(indices, matched_names) if name == body_name]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected exactly one body named {body_name!r}; find_bodies returned {matched_names}"
        )
    return int(exact[0][0])


class FixedJointWrenchReader:
    """Read an uncancelled incoming reaction by its downstream child body.

    Fixed joints are not movable-joint DOFs.  Isaac Lab exposes their reaction
    on ``robot.data.body_incoming_joint_wrench_b`` indexed by the *child body*.
    The asset must therefore preserve fixed joints (URDF conversion with
    ``merge_fixed_joints=False``).

    This task models an F/T sensor with no cancellation.  The reader therefore
    exposes the reaction exactly as simulated and performs only the rigid
    F-to-C coordinate/moment-reference transform requested by the caller.
    """

    def __init__(self, robot: Any, child_body_name: str) -> None:
        self.robot = robot
        self.child_body_name = child_body_name
        self.child_body_index = _exact_body_index(robot, child_body_name)

    def read_raw_f(self) -> torch.Tensor:
        all_wrenches = self.robot.data.body_incoming_joint_wrench_b
        if not isinstance(all_wrenches, torch.Tensor) or all_wrenches.ndim != 3 or all_wrenches.shape[-1] != 6:
            raise ValueError(
                "body_incoming_joint_wrench_b must have shape (num_envs, num_bodies, 6)"
            )
        if self.child_body_index >= all_wrenches.shape[1]:
            raise RuntimeError(
                f"Child body index {self.child_body_index} exceeds wrench body dimension "
                f"{all_wrenches.shape[1]}"
            )
        raw_f = all_wrenches[:, self.child_body_index, :]
        if not torch.isfinite(raw_f).all():
            raise RuntimeError("Incoming fixed-joint wrench contains non-finite values")
        return raw_f

    def read(
        self,
        rotation_c_from_f: torch.Tensor | None = None,
        r_c_to_f_c: torch.Tensor | None = None,
    ) -> FixedJointWrenchSample:
        raw_f = self.read_raw_f()
        measured_f = raw_f

        if rotation_c_from_f is None and r_c_to_f_c is None:
            measured_c = measured_f
        elif rotation_c_from_f is None or r_c_to_f_c is None:
            raise ValueError("rotation_c_from_f and r_c_to_f_c must be provided together")
        else:
            measured_c = transform_wrench_f_to_c(
                measured_f, rotation_c_from_f, r_c_to_f_c
            )
        return FixedJointWrenchSample(
            raw_f=raw_f,
            measured_f=measured_f,
            measured_c=measured_c,
        )


@dataclass(frozen=True)
class BoardContactSample:
    """Robot-board normal contact force, preserving one value per robot link."""

    pair_forces_w: torch.Tensor
    per_link_magnitudes_n: torch.Tensor
    total_magnitude_n: torch.Tensor


def robot_board_contact_forces(
    board_sensor: Any,
    *,
    board_body_index: int | None = None,
    robot_filter_indices: Sequence[int] | torch.Tensor | None = None,
) -> BoardContactSample:
    """Aggregate a board ContactSensor's one-to-many force matrix.

    Configure the sensor filters to contain robot bodies only; the object's
    normal support force must not be one of these columns.  For each robot-link
    filter this function first takes the vector norm, then sums magnitudes.  It
    intentionally does *not* norm the vector sum, which could cancel contacts
    acting in opposite directions.

    Expected Isaac Lab shape is ``(N, board_bodies, robot_filters, 3)``.  The
    shorthand ``(N, robot_filters, 3)`` is also accepted for a single board
    body.  ``force_matrix_w`` contains normal forces, not friction forces.
    """

    matrix = board_sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError(
            "Board ContactSensor has no force_matrix_w; configure filter_prim_paths_expr for robot links"
        )
    if not isinstance(matrix, torch.Tensor) or matrix.shape[-1:] != (3,) or matrix.ndim not in (3, 4):
        raise ValueError(
            "Board force_matrix_w must have shape (N, filters, 3) or (N, bodies, filters, 3)"
        )
    if not torch.isfinite(matrix).all():
        raise RuntimeError("Board force matrix contains non-finite values")
    if matrix.ndim == 3:
        pair_forces_w = matrix.unsqueeze(1)
    elif board_body_index is None:
        pair_forces_w = matrix
    else:
        if board_body_index < 0 or board_body_index >= matrix.shape[1]:
            raise IndexError(f"board_body_index {board_body_index} is out of range")
        pair_forces_w = matrix[:, board_body_index : board_body_index + 1]

    if robot_filter_indices is not None:
        indices = torch.as_tensor(
            robot_filter_indices, dtype=torch.long, device=pair_forces_w.device
        )
        pair_forces_w = torch.index_select(pair_forces_w, dim=2, index=indices)

    # Pair magnitudes: (N, board_bodies, robot_filters).  Sum across board
    # bodies but retain robot-link filters, then sum links for the termination
    # scalar required by the task guide.
    per_link_magnitudes_n = torch.linalg.vector_norm(pair_forces_w, dim=-1).sum(dim=1)
    total_magnitude_n = per_link_magnitudes_n.sum(dim=-1)
    return BoardContactSample(pair_forces_w, per_link_magnitudes_n, total_magnitude_n)


@dataclass(frozen=True)
class ResultantNormal:
    """Force-weighted hand-object contact normal and validity mask."""

    force_on_object_w: torch.Tensor
    magnitude_n: torch.Tensor
    normal_w: torch.Tensor
    valid: torch.Tensor


def hand_object_resultant_normal(
    normal_force_vectors_w: torch.Tensor,
    *,
    input_forces_on_hand: bool = True,
    contact_dims: int | Sequence[int] | None = None,
    min_resultant_force_n: float = 1.0e-6,
) -> ResultantNormal:
    """Return the force-weighted normal exerted by the hand on the object.

    Args:
        normal_force_vectors_w: Per-contact or per-link normal force vectors in
            world coordinates.  Shape is conventionally ``(N, contacts, 3)``
            or ``(N, hand_bodies, object_filters, 3)``.
        input_forces_on_hand: ContactSensor forces are commonly the reaction on
            the observed hand body.  They are negated by default to obtain the
            direction exerted on the object, which is the reward convention.
        contact_dims: Dimensions to sum.  By default all dimensions between the
            leading environment dimension and xyz are reduced.
        min_resultant_force_n: Below this positive threshold the direction is
            undefined; ``normal_w`` is zero and ``valid`` is false.
    """

    vectors = _finite_force_vectors(normal_force_vectors_w, "normal_force_vectors_w")
    threshold = _positive_finite(min_resultant_force_n, "min_resultant_force_n")
    if input_forces_on_hand:
        vectors = -vectors
    if contact_dims is None:
        dims = tuple(range(1, vectors.ndim - 1))
    elif isinstance(contact_dims, int):
        dims = (contact_dims,)
    else:
        dims = tuple(contact_dims)
    if any(dim == vectors.ndim - 1 or dim == -1 for dim in dims):
        raise ValueError("contact_dims must not reduce the xyz dimension")
    force_on_object_w = vectors.sum(dim=dims) if dims else vectors
    magnitude_n = torch.linalg.vector_norm(force_on_object_w, dim=-1)
    valid = magnitude_n >= threshold
    normal_w = torch.where(
        valid.unsqueeze(-1),
        force_on_object_w / magnitude_n.clamp_min(threshold).unsqueeze(-1),
        torch.zeros_like(force_on_object_w),
    )
    return ResultantNormal(force_on_object_w, magnitude_n, normal_w, valid)


def weighted_hand_object_resultant_normal(
    contact_normals_w: torch.Tensor,
    normal_force_magnitudes_n: torch.Tensor,
    *,
    normals_are_forces_on_hand: bool = True,
    contact_dims: int | Sequence[int] | None = None,
    min_resultant_force_n: float = 1.0e-6,
) -> ResultantNormal:
    """Convenience wrapper for separate unit normals and positive magnitudes."""

    normals = _finite_force_vectors(contact_normals_w, "contact_normals_w")
    if not isinstance(normal_force_magnitudes_n, torch.Tensor):
        raise TypeError("normal_force_magnitudes_n must be a torch.Tensor")
    if not torch.isfinite(normal_force_magnitudes_n).all() or (normal_force_magnitudes_n < 0).any():
        raise ValueError("normal_force_magnitudes_n must be finite and non-negative")
    if normal_force_magnitudes_n.shape != normals.shape[:-1]:
        raise ValueError(
            "normal_force_magnitudes_n shape must equal contact_normals_w without xyz"
        )
    lengths = torch.linalg.vector_norm(normals, dim=-1)
    nonzero = normal_force_magnitudes_n > 0
    if not torch.allclose(lengths[nonzero], torch.ones_like(lengths[nonzero]), atol=1.0e-4, rtol=1.0e-4):
        raise ValueError("Nonzero-force contact normals must be unit length")
    vectors = normals * normal_force_magnitudes_n.unsqueeze(-1)
    return hand_object_resultant_normal(
        vectors,
        input_forces_on_hand=normals_are_forces_on_hand,
        contact_dims=contact_dims,
        min_resultant_force_n=min_resultant_force_n,
    )


__all__ = [
    "BilateralTactileReader",
    "BilateralTactileSample",
    "BoardContactSample",
    "DORSAL_CHANNEL_NAMES",
    "DORSAL_PARENT_BODY_NAMES",
    "DORSAL_PARENT_BY_CHANNEL",
    "DORSAL_PARENT_MAP",
    "DORSAL_SENSOR_NAMES",
    "FixedJointWrenchReader",
    "FixedJointWrenchSample",
    "PALM_CHANNEL_NAMES",
    "PALM_SENSOR_NAMES",
    "ResultantNormal",
    "TACTILE_CHANNEL_COUNT_PER_SIDE",
    "contact_magnitudes_and_bits",
    "hand_object_resultant_normal",
    "robot_board_contact_forces",
    "transform_wrench_f_to_c",
    "weighted_hand_object_resultant_normal",
]
