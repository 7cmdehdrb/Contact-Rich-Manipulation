"""Tensor-only implicit Cartesian impedance with optional measured-load prediction."""

from __future__ import annotations

import torch

from .action_math import quaternion_conjugate, quaternion_rotate


def cube_reaction_wrench_b(forces_on_cube_w, pad_centers_w, control_point_w, base_quaternion_w):
    """Equal/opposite pad reactions about C, expressed in robot-base axes.

    Cube–Table support is absent from the supplied Cube–palm force matrix.
    Do not rotate this wrench with C's orientation: the controller Jacobian
    and pose errors use the robot-base Cartesian axes.
    """

    reaction = -torch.nan_to_num(forces_on_cube_w, nan=0., posinf=0., neginf=0.)
    lever = pad_centers_w - control_point_w[:, None]
    force = reaction.sum(1)
    moment = torch.linalg.cross(lever, reaction, dim=-1).sum(1)
    inverse_base = quaternion_conjugate(base_quaternion_w)
    return torch.cat((quaternion_rotate(inverse_base, force), quaternion_rotate(inverse_base, moment)), -1)


def implicit_cartesian_impedance_efforts(
    jacobian: torch.Tensor,
    joint_mass: torch.Tensor,
    pose_error: torch.Tensor,
    twist: torch.Tensor,
    stiffness: torch.Tensor,
    damping_ratio: torch.Tensor,
    physics_dt: float,
    gravity: torch.Tensor | None = None,
    external_wrench: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stabilize a configured spring/damper with an implicit timestep update.

    The configured stiffness is a force/position gain, rather than an
    acceleration gain multiplied by the operational inertia. Damping is
    scaled to operational mass. The implicit update bounds stiff rotational
    responses at finite simulator timesteps. Without measured external load,
    it softens the effective static gain: for a decoupled axis it is
    ``M*K/(M+dt*D+dt**2*K)``. Configured 200 then does not imply an exact 200
    N/m or Nm/rad equilibrium gain at dt=10 ms, especially for tiny inertia.

    ``external_wrench`` is the measured force/moment acting ON the robot,
    expressed at the same point and in the same task frame as pose_error and
    twist. Including that load in the implicit prediction restores the
    configured static equilibrium ``K*error + external_wrench = 0`` when
    the measurement matches the actual load. Unmeasured load still softens
    the response. Gravity remains a separate joint-effort compensation;
    do not include the same gravity load again in external_wrench. The caller
    applies the robot's joint effort limits to the returned efforts.
    """

    inverse_mass_jt = torch.linalg.solve(joint_mass, jacobian.mT)
    inverse_task_mass = jacobian @ inverse_mass_jt
    inverse_task_mass = .5 * (inverse_task_mass + inverse_task_mass.mT)
    identity = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype).expand_as(inverse_task_mass)
    # The small regularizer remains finite near a kinematic singularity.
    task_mass = torch.linalg.solve(inverse_task_mass + 1.e-6 * identity, identity)
    task_mass = .5 * (task_mass + task_mass.mT)
    damping = 2. * damping_ratio * torch.sqrt(
        stiffness * torch.diagonal(task_mass, dim1=-2, dim2=-1).clamp_min(1.e-8)
    )
    effective_mass = task_mass + torch.diag_embed(physics_dt * damping + physics_dt**2 * stiffness)
    forcing = stiffness * pose_error - (damping + physics_dt * stiffness) * twist
    if external_wrench is not None:
        forcing = forcing + external_wrench
    acceleration = torch.linalg.solve(effective_mass, forcing.unsqueeze(-1))
    wrench = task_mass @ acceleration
    if external_wrench is not None:
        # Include the measured contact reaction in the implicit prediction.
        # At equilibrium this removes timestep-dependent spring softening:
        # K*error + external = 0, controller_wrench = -external.
        wrench = wrench - external_wrench.unsqueeze(-1)
    efforts = (jacobian.mT @ wrench).squeeze(-1)
    return efforts if gravity is None else efforts + gravity


__all__ = ["implicit_cartesian_impedance_efforts", "cube_reaction_wrench_b"]
