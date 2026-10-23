"""Physical sensor math: uncancelled fixed-joint reaction and contact bits.

Only torch is required here, so body ordering and wrench transport can be
verified without launching Isaac Sim. Reaction forces include hand weight;
there is no gravity cancellation, tare, filtering, or reward input."""

from __future__ import annotations

from dataclasses import dataclass
import math as math
import torch as torch

def contact_magnitudes_and_bits(forces_w, threshold_n=0.05):
    threshold_n = float(threshold_n)
    if not math.isfinite(threshold_n) or threshold_n <= 0.0:
        raise ValueError("threshold_n must be positive and finite")
    if forces_w.ndim < 2 or forces_w.shape[-1] != 3:
        raise ValueError("forces_w must have shape (..., channels, 3)")
    if not torch.isfinite(forces_w).all():
        raise RuntimeError("Contact force contains non-finite values")
    magnitude = torch.linalg.vector_norm(forces_w, dim=-1)
    return magnitude, (magnitude >= threshold_n).to(torch.uint8)


def transform_wrench_f_to_c(wrench_f, rotation_c_from_f, r_c_to_f_c):
    """Rotate F into C and shift moment by r(C→F) × force, in C axes."""
    if wrench_f.shape[-1:] != (6,) or rotation_c_from_f.shape[-2:] != (3, 3):
        raise ValueError("Expected (..., 6) wrench and (..., 3, 3) rotation")
    if r_c_to_f_c.shape[-1:] != (3,):
        raise ValueError("Moment arm must have shape (..., 3)")
    if not all(torch.isfinite(tensor).all() for tensor in (wrench_f, rotation_c_from_f, r_c_to_f_c)):
        raise RuntimeError("Wrench transform contains non-finite values")
    force_c = (rotation_c_from_f @ wrench_f[..., :3, None]).squeeze(-1)
    moment_c = (rotation_c_from_f @ wrench_f[..., 3:, None]).squeeze(-1)
    arm = torch.broadcast_to(r_c_to_f_c, force_c.shape)
    return torch.cat((force_c, moment_c + torch.cross(arm, force_c, dim=-1)), dim=-1)


@dataclass(frozen=True)
class FixedJointWrenchSample:
    raw_f: torch.Tensor
    measured_f: torch.Tensor
    measured_c: torch.Tensor


class FixedJointWrenchReader:
    """The preserved mount reaction is indexed by its downstream child body."""

    def __init__(self, robot, child_body_name="inspire_base_link"):
        self.robot = robot
        names = tuple(robot.body_names)
        if names.count(child_body_name) != 1:
            raise RuntimeError(f"Expected exactly one {child_body_name!r}; got {names}")
        self.child_body_index = names.index(child_body_name)

    def read_raw_f(self, valid_env_mask=None):
        wrenches = self.robot.data.body_incoming_joint_wrench_b
        if wrenches.ndim != 3 or wrenches.shape[-1] != 6:
            raise ValueError("body_incoming_joint_wrench_b must have shape (envs, bodies, 6)")
        if self.child_body_index >= wrenches.shape[1]:
            raise RuntimeError("Wrench body dimension does not include the mount child body")
        raw = wrenches[:, self.child_body_index]
        if valid_env_mask is not None:
            # Reset samples may be stale or invalid; exclude those rows before validation.
            raw = torch.where(valid_env_mask[:, None], raw, torch.zeros_like(raw))
        if not torch.isfinite(raw).all():
            raise RuntimeError("Fixed-joint wrench contains non-finite values")
        return raw

    def read(self, rotation_c_from_f=None, r_c_to_f_c=None, valid_env_mask=None):
        raw = self.read_raw_f(valid_env_mask)
        if rotation_c_from_f is None and r_c_to_f_c is None:
            measured_c = raw
        elif rotation_c_from_f is None or r_c_to_f_c is None:
            raise ValueError("Rotation and moment arm must be provided together")
        else:
            measured_c = transform_wrench_f_to_c(raw, rotation_c_from_f, r_c_to_f_c)
        return FixedJointWrenchSample(raw_f=raw, measured_f=raw, measured_c=measured_c)
