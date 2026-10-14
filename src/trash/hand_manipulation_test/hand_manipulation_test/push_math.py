"""Pure geometry for horizontal, table-supported Cube pushing."""

from __future__ import annotations

import math

import torch

from .contact_math import palm_facing_rotation


def push_direction_from_angle(angle_rad: torch.Tensor) -> torch.Tensor:
    """Use world +Y for right (0), world -Y for left (pi).

    The robot base has yaw pi, so these directions are base -Y and +Y.
    Positive angles turn counterclockwise about world +Z.
    """

    return torch.stack((-torch.sin(angle_rad), torch.cos(angle_rad), torch.zeros_like(angle_rad)), dim=-1)


def push_palm_rotation(direction_w: torch.Tensor) -> torch.Tensor:
    """Keep H +Y horizontal along the push and H +Z toward world -X."""

    horizontal = direction_w.clone()
    horizontal[..., 2] = 0.0
    if bool((horizontal.norm(dim=-1) <= torch.finfo(horizontal.dtype).eps).any()):
        raise ValueError("A push direction must have a nonzero horizontal component")
    return palm_facing_rotation(horizontal)


def push_palm_start_pose(
    cube_position_w: torch.Tensor,
    direction_w: torch.Tensor,
    task_offset: torch.Tensor,
    palm_reference_h: torch.Tensor,
    control_offset_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place the actual exposed pad face behind the Cube and above its centre.

    ``task_offset`` is pad-face backoff, horizontal tangent, and world-up
    height. The control point is derived from that physical pad reference.
    """

    rotation = push_palm_rotation(direction_w)
    direction = rotation[..., :, 1]
    tangent = torch.stack((-direction[..., 1], direction[..., 0], torch.zeros_like(direction[..., 0])), dim=-1)
    up = torch.zeros_like(direction)
    up[..., 2] = 1.0
    palm_position = (
        cube_position_w - task_offset[..., :1] * direction
        + task_offset[..., 1:2] * tangent + task_offset[..., 2:3] * up
    )
    delta = control_offset_h - palm_reference_h
    control_position = palm_position + (rotation @ delta.unsqueeze(-1)).squeeze(-1)
    return control_position, rotation, palm_position


def cube_circumsphere_radius(cube_size: float) -> float:
    """Bound any Cube orientation's horizontal footprint by its 3-D radius."""

    if not math.isfinite(cube_size) or cube_size <= 0.0:
        raise ValueError("Cube size must be finite and positive")
    return math.sqrt(3.0) * cube_size * 0.5


def push_start_xy_bounds(
    object_low: torch.Tensor,
    object_high: torch.Tensor,
    table_center_xy: torch.Tensor,
    table_size_xy: torch.Tensor,
    displacement_xy: torch.Tensor,
    cube_size: float,
    path_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect the spawn box with safe start and end boxes for one command."""

    if not math.isfinite(path_margin) or path_margin < 0.0:
        raise ValueError("Path margin must be finite and nonnegative")
    radius = cube_circumsphere_radius(cube_size) + path_margin
    surface_low = table_center_xy - table_size_xy * 0.5 + radius
    surface_high = table_center_xy + table_size_xy * 0.5 - radius
    low = torch.maximum(torch.maximum(object_low, surface_low), surface_low - displacement_xy)
    high = torch.minimum(torch.minimum(object_high, surface_high), surface_high - displacement_xy)
    return low, high


def push_path_within_table(
    start_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    table_center_xy: torch.Tensor,
    table_size_xy: torch.Tensor,
    cube_size: float,
    path_margin: float,
) -> torch.Tensor:
    """Certify the entire straight path in an orientation-safe convex rectangle."""

    radius = cube_circumsphere_radius(cube_size) + path_margin
    low = table_center_xy - table_size_xy * 0.5 + radius
    high = table_center_xy + table_size_xy * 0.5 - radius
    return ((start_xy >= low) & (start_xy <= high) & (goal_xy >= low) & (goal_xy <= high)).all(dim=-1)


def cube_planar_support_radius(direction_w: torch.Tensor, cube_size: float) -> torch.Tensor:
    """Distance to the supporting plane of an upright, axis-aligned Cube."""

    cube_circumsphere_radius(cube_size)
    normal = push_palm_rotation(direction_w)[..., :, 1]
    return 0.5 * cube_size * normal[..., :2].abs().sum(dim=-1)


__all__ = [
    "push_direction_from_angle", "push_palm_rotation", "push_palm_start_pose", "cube_circumsphere_radius",
    "push_start_xy_bounds", "push_path_within_table", "cube_planar_support_radius",
]
