"""Dynamic tool-load cancellation for the preserved Axia80 measurement joint.

The feasibility experiments subtract a measured, static tool-only tare.  A moving
articulated hand instead needs its current gravity *and* inertial support wrench.
This module computes that tool-only reference from every downstream rigid body.
The residual keeps the sensor's parent-on-child reaction sign: an external force
on the hand appears with the opposite sign in the corrected sensor reading.

PhysX reports a step-averaged joint reaction, so this is a model-based estimate
of external wrench, not an exact continuous-time contact-force measurement.
Solver/discretization residuals and unmodelled body damping can remain."""

from __future__ import annotations

from ...assets.robot import HAND_CONTACT_BODY_NAMES
from .sensors import FixedJointWrenchReader
from dataclasses import dataclass
from isaaclab.utils.math import matrix_from_quat
import torch as torch

def support_wrench_world(
    masses: torch.Tensor,
    inertias_b: torch.Tensor,
    body_quat_w: torch.Tensor,
    com_pos_w: torch.Tensor,
    com_vel_w: torch.Tensor,
    com_acc_w: torch.Tensor,
    reference_pos_w: torch.Tensor,
    gravity_w: torch.Tensor,
) -> torch.Tensor:
    """Sum Newton--Euler support loads about ``reference_pos_w`` in world axes.

    ``inertias_b`` are COM inertia tensors expressed in each *link* frame,
    matching PhysX ``get_inertias``.  Rotating these by COM/principal-axis
    quaternions would rotate the inertia twice when the COM frame is rotated.
    All body arrays have leading shape ``(envs, downstream_bodies)``.
    """
    rotations = matrix_from_quat(body_quat_w)
    inertias_w = rotations @ inertias_b @ rotations.transpose(-1, -2)
    forces_w = masses[..., None] * (com_acc_w[..., :3] - gravity_w)
    omega_w = com_vel_w[..., 3:]
    momentum_w = (inertias_w @ omega_w[..., None]).squeeze(-1)
    moments_w = (
        (inertias_w @ com_acc_w[..., 3:, None]).squeeze(-1)
        + torch.cross(omega_w, momentum_w, dim=-1)
        + torch.cross(com_pos_w - reference_pos_w[:, None], forces_w, dim=-1)
    )
    return torch.cat((forces_w.sum(dim=1), moments_w.sum(dim=1)), dim=-1)


def rotate_wrench(wrench: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate force and moment without changing their reference point."""
    return torch.cat(
        (
            (rotation @ wrench[..., :3, None]).squeeze(-1),
            (rotation @ wrench[..., 3:, None]).squeeze(-1),
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class PureWrenchSample:
    """N/Nm values, all about the EEF point; ``world`` denotes axes only."""

    raw_eef: torch.Tensor
    support_eef: torch.Tensor
    pure_eef: torch.Tensor
    raw_world: torch.Tensor
    support_world: torch.Tensor
    pure_world: torch.Tensor


class PureWrenchReader:
    """Read the mount reaction and remove downstream gravity/inertia.

    The generated ``axia80_measurement_joint`` has its child joint frame at
    ``inspire_base_link`` with identity local pose.  PhysX Tensor API reports
    incoming force in that child *joint* frame, not the Axia housing center
    (12.7 mm upstream).  Use the child pose for both axes and moment transport.

    ``body_com_acc_w`` is the acceleration from the same latest physics step as
    the raw joint reaction; no control-step velocity difference is substituted.
    Call only after ``scene.update``.  Observations/rewards should share a cached
    sample.  Reset rows are masked before validation, and no fixed/learned tare
    is collected, so a real contact at reset cannot be learned away.

    Link mass/inertia are static for this task and are cached at construction.
    Call ``refresh_mass_properties`` after any future hand mass randomization.
    """

    def __init__(self, robot, gravity_w=(0.0, 0.0, -9.81)):
        self.robot = robot
        self.raw_reader = FixedJointWrenchReader(robot)
        names = tuple(robot.body_names)
        missing = [name for name in HAND_CONTACT_BODY_NAMES if names.count(name) != 1]
        if missing:
            raise RuntimeError(f"Expected exactly one of every downstream hand body; invalid: {missing}")
        self.body_ids = [names.index(name) for name in HAND_CONTACT_BODY_NAMES]
        self.child_body_index = self.raw_reader.child_body_index
        self.gravity_w = torch.as_tensor(gravity_w, device=robot.device, dtype=torch.float32)
        if self.gravity_w.shape != (3,) or not torch.isfinite(self.gravity_w).all():
            raise ValueError("gravity_w must be a finite three-vector")
        self.refresh_mass_properties()

    def refresh_mass_properties(self):
        """Refresh actual PhysX hand properties, including fixed tactile pads."""
        view = self.robot.root_physx_view
        self.masses = view.get_masses()[:, self.body_ids].to(self.robot.device).clone()
        # PhysX returns column-major 3x3 tensors, COM origin and body-link axes.
        self.inertias_b = (
            view.get_inertias()[:, self.body_ids]
            .to(self.robot.device)
            .reshape(*self.masses.shape, 3, 3)
            .transpose(-1, -2)
            .clone()
        )
        if not torch.isfinite(self.masses).all() or not torch.isfinite(self.inertias_b).all():
            raise RuntimeError("Downstream hand mass/inertia contains non-finite values")
        if bool((self.masses <= 0).any()):
            raise ValueError("Downstream hand masses must be positive")

    def read(self, eef_pos_w, eef_quat_w, valid_env_mask=None) -> PureWrenchSample:
        """Return one compensated sample; stale/reset rows contain six zeros."""
        data = self.robot.data
        count = self.masses.shape[0]
        valid = (
            torch.ones(count, dtype=torch.bool, device=self.masses.device)
            if valid_env_mask is None else valid_env_mask
        )
        if valid.shape != (count,) or valid.dtype != torch.bool:
            raise ValueError("valid_env_mask must be bool with shape (envs,)")
        zeros = self.masses.new_zeros((count, 6))
        if not bool(valid.any()):
            return PureWrenchSample(*(zeros.clone() for _ in range(6)))

        def mask(value, *, quaternion=False):
            default = torch.zeros_like(value)
            if quaternion:
                default[..., 0] = 1.0
            return torch.where(valid.reshape(count, *((1,) * (value.ndim - 1))), value, default)

        raw_joint = self.raw_reader.read_raw_f(valid)
        child_pos = mask(data.body_link_pos_w[:, self.child_body_index])
        child_quat = mask(data.body_link_quat_w[:, self.child_body_index], quaternion=True)
        eef_pos = mask(eef_pos_w)
        eef_quat = mask(eef_quat_w, quaternion=True)
        com_pos = mask(data.body_com_pos_w[:, self.body_ids])
        com_vel = mask(data.body_com_vel_w[:, self.body_ids])
        com_acc = mask(data.body_com_acc_w[:, self.body_ids])
        body_quat = mask(data.body_link_quat_w[:, self.body_ids], quaternion=True)
        tensors = (child_pos, child_quat, eef_pos, eef_quat, com_pos, com_vel, com_acc, body_quat)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError("Wrench compensation state contains non-finite values")

        raw_world = rotate_wrench(raw_joint, matrix_from_quat(child_quat))
        raw_world = torch.cat(
            (
                raw_world[:, :3],
                raw_world[:, 3:] + torch.cross(child_pos - eef_pos, raw_world[:, :3], dim=-1),
            ),
            dim=-1,
        )
        support_world = support_wrench_world(
            self.masses, self.inertias_b, body_quat, com_pos, com_vel, com_acc,
            eef_pos, self.gravity_w,
        )
        support_world = mask(support_world)
        pure_world = raw_world - support_world
        eef_from_world = matrix_from_quat(eef_quat).transpose(-1, -2)
        return PureWrenchSample(
            raw_eef=rotate_wrench(raw_world, eef_from_world),
            support_eef=rotate_wrench(support_world, eef_from_world),
            pure_eef=rotate_wrench(pure_world, eef_from_world),
            raw_world=raw_world,
            support_world=support_world,
            pure_world=pure_world,
        )
