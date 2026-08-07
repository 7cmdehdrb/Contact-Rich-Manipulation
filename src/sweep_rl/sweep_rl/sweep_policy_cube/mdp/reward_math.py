"""Pure tensor kernels matching the fixed Sweep-Policy rewards."""

from __future__ import annotations

import torch


def fixed_policy_hand_target(
    target_position_w: torch.Tensor,
    target_width: float,
    sweep_direction_y: torch.Tensor,
) -> torch.Tensor:
    """Return the Sweep-Policy hand target on the upstream side of the object."""

    hand_target = target_position_w.clone()
    hand_target[:, 1] -= target_width * torch.sign(sweep_direction_y)
    hand_target[:, 2] += 0.03
    return hand_target


def position_error_tanh_kernel(
    distance: torch.Tensor, std: float
) -> torch.Tensor:
    """Isaac Lab Reach fine-position kernel: ``1 - tanh(distance / std)``."""

    return 1.0 - torch.tanh(distance / std)


def pushing_reward_kernel(
    distance: torch.Tensor,
    contact_distance: torch.Tensor,
    wrist_y_error: torch.Tensor,
    target_y_velocity: torch.Tensor,
) -> torch.Tensor:
    # Match example/Sweep-Policy: in the upright pushing pose the wrist and the
    # table-side finger contact share approximately the same world-Y position.
    zeta_m = (contact_distance < 0.04) & (wrist_y_error < 0.04)
    velocity_reward = torch.where(
        target_y_velocity > 0.05,
        torch.where(target_y_velocity < 0.10, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * (1.0 - distance / 0.18),
        zeta_m.float() * ((1.0 - distance / 0.18) + velocity_reward),
    )


def axis_alignment_error_kernel(alignment: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular error between two directed unit axes.

    The input is the dot product of the axes.  A value of ``1`` therefore has
    zero error, orthogonal axes have ``pi / 2`` error, and antiparallel axes
    have ``pi`` error.
    """

    return torch.acos(torch.clamp(alignment, min=-1.0, max=1.0))


def ee_y_shelf_z_alignment_kernel(
    ee_rotation: torch.Tensor, shelf_rotation: torch.Tensor
) -> torch.Tensor:
    """Return ``dot(EE local-y in world, shelf local-z in world)``.

    Rotation matrices map local column vectors into the parent frame, so their
    local basis axes are stored in columns rather than rows.
    """

    ee_y_axis_w = ee_rotation[..., :, 1]
    shelf_z_axis_w = shelf_rotation[..., :, 2]
    return torch.sum(ee_y_axis_w * shelf_z_axis_w, dim=-1)
