"""Pure tensor geometry for object-relative palm-facing initialization."""

from __future__ import annotations

import torch


def palm_facing_rotation(normal_w: torch.Tensor) -> torch.Tensor:
    """Choose H +Y toward the object and H +Z toward projected world -X."""

    hand_y = torch.nn.functional.normalize(normal_w, dim=-1)
    preferred = torch.zeros_like(hand_y)
    preferred[..., 0] = -1.0
    hand_z = preferred - (preferred * hand_y).sum(dim=-1, keepdim=True) * hand_y
    fallback = torch.zeros_like(hand_y)
    fallback[..., 2] = 1.0
    fallback -= (fallback * hand_y).sum(dim=-1, keepdim=True) * hand_y
    hand_z = torch.where(hand_z.norm(dim=-1, keepdim=True) < 1.0e-6, fallback, hand_z)
    hand_z = torch.nn.functional.normalize(hand_z, dim=-1)
    hand_x = torch.linalg.cross(hand_y, hand_z, dim=-1)
    return torch.stack((hand_x, hand_y, hand_z), dim=-1)


def palm_facing_start_pose(
    cube_position_w: torch.Tensor,
    direction_sign: torch.Tensor,
    task_offset: torch.Tensor,
    palm_reference_h: torch.Tensor,
    control_offset_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate about the actual pad face while retaining its sampled position.

    ``task_offset`` contains backoff, shelf tangent, and world-up height.
    The original planar pose establishes the palm face position; exact 3-D
    facing then rotates the hand about that face instead of its wrist origin.
    Returns the C position, H rotation matrix, and palm face world position.
    """

    sign = direction_sign.reshape(-1)
    hand_y = torch.zeros_like(cube_position_w)
    hand_y[:, 1] = sign
    hand_x = torch.zeros_like(hand_y)
    hand_x[:, 2] = sign
    hand_z = torch.linalg.cross(hand_x, hand_y, dim=-1)
    nominal_rotation = torch.stack((hand_x, hand_y, hand_z), dim=-1)
    tangent = torch.stack((-sign, torch.zeros_like(sign), torch.zeros_like(sign)), dim=-1)
    up = torch.zeros_like(hand_y)
    up[:, 2] = 1.0
    nominal_c = (
        cube_position_w - task_offset[:, :1] * hand_y
        + task_offset[:, 1:2] * tangent + task_offset[:, 2:3] * up
    )
    reference_delta = palm_reference_h - control_offset_h
    palm_position = nominal_c + (nominal_rotation @ reference_delta.expand(len(sign), -1).unsqueeze(-1)).squeeze(-1)
    rotation = palm_facing_rotation(cube_position_w - palm_position)
    c_position = palm_position + (rotation @ (-reference_delta).expand(len(sign), -1).unsqueeze(-1)).squeeze(-1)
    return c_position, rotation, palm_position


def oriented_box_overlap(
    center_a: torch.Tensor,
    rotation_a: torch.Tensor,
    half_extent_a: torch.Tensor,
    center_b: torch.Tensor,
    rotation_b: torch.Tensor,
    half_extent_b: torch.Tensor,
) -> torch.Tensor:
    """Conservative complete 15-axis SAT; touching boxes count as overlap."""

    rotation = rotation_a.transpose(-1, -2) @ rotation_b
    translation = (rotation_a.transpose(-1, -2) @ (center_b - center_a).unsqueeze(-1)).squeeze(-1)
    absolute = rotation.abs() + 8.0 * torch.finfo(rotation.dtype).eps
    separated = (translation.abs() > half_extent_a + (absolute * half_extent_b.unsqueeze(-2)).sum(-1)).any(-1)
    translation_b = (translation.unsqueeze(-2) @ rotation).squeeze(-2)
    separated |= (translation_b.abs() > half_extent_b + (absolute * half_extent_a.unsqueeze(-1)).sum(-2)).any(-1)
    for axis_a in range(3):
        next_a, last_a = (axis_a + 1) % 3, (axis_a + 2) % 3
        for axis_b in range(3):
            next_b, last_b = (axis_b + 1) % 3, (axis_b + 2) % 3
            distance = (translation[..., last_a] * rotation[..., next_a, axis_b]
                        - translation[..., next_a] * rotation[..., last_a, axis_b]).abs()
            radius_a = (half_extent_a[..., next_a] * absolute[..., last_a, axis_b]
                        + half_extent_a[..., last_a] * absolute[..., next_a, axis_b])
            radius_b = (half_extent_b[..., next_b] * absolute[..., axis_a, last_b]
                        + half_extent_b[..., last_b] * absolute[..., axis_a, next_b])
            separated |= distance > radius_a + radius_b
    return ~separated
