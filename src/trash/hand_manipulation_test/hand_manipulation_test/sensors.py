"""Simulator-independent adapters for palmar tactile and uncancelled wrist wrench."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


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
class PalmTactileSample:
    """Physical palmar-pad data in the canonical seventeen-channel order."""

    forces_w: torch.Tensor
    magnitudes_n: torch.Tensor
    bits: torch.Tensor


class PalmTactileReader:
    """Read all seventeen physical palmar pads without dorsal channels."""

    def __init__(
        self,
        palm_sensor: Any,
        *,
        threshold_n: float = 0.05,
        palm_channel_names: Sequence[str] = PALM_CHANNEL_NAMES,
    ) -> None:
        if len(palm_channel_names) != 17:
            raise ValueError("palm_channel_names must define exactly seventeen channels")
        self.palm_sensor = palm_sensor
        self.threshold_n = _positive_finite(threshold_n, "threshold_n")
        self._indices = _name_indices(
            palm_sensor.body_names,
            palm_channel_names,
            require_exact_set=True,
            source="palm ContactSensor",
        )
        self._index_tensor: torch.Tensor | None = None

    def read(self) -> PalmTactileSample:
        forces = _finite_force_vectors(self.palm_sensor.data.net_forces_w, "palm net_forces_w")
        if forces.ndim != 3 or forces.shape[1] != 17:
            raise ValueError("palm net_forces_w must have shape (num_envs, 17, 3)")
        if self._index_tensor is None or self._index_tensor.device != forces.device:
            self._index_tensor = torch.tensor(self._indices, dtype=torch.long, device=forces.device)
        canonical_forces = torch.index_select(forces, dim=1, index=self._index_tensor)
        magnitudes, bits = contact_magnitudes_and_bits(canonical_forces, self.threshold_n)
        return PalmTactileSample(canonical_forces, magnitudes, bits)


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


