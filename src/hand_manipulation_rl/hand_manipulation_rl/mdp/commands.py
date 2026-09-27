"""Sweep-command geometry and sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


Tensor = torch.Tensor

SHELF_LEFT_ANGLE = -math.pi / 2.0
SHELF_RIGHT_ANGLE = math.pi / 2.0
LATERAL_ANGLE_TOLERANCE = 1.0e-6


def wrap_s_angle(angle: Tensor | float) -> Tensor:
    """Wrap an S-frame angle to ``[-pi/2, 3*pi/2)``."""

    value = torch.as_tensor(angle)
    return torch.remainder(value + math.pi / 2.0, 2.0 * math.pi) - math.pi / 2.0


def circular_angle_difference(first: Tensor | float, second: Tensor | float) -> Tensor:
    """Return the signed shortest difference ``first - second`` in radians."""

    first_tensor = torch.as_tensor(first)
    second_tensor = torch.as_tensor(second, dtype=first_tensor.dtype, device=first_tensor.device)
    return torch.remainder(first_tensor - second_tensor + math.pi, 2.0 * math.pi) - math.pi


def is_allowed_sweep_angle(angle: Tensor | float) -> Tensor:
    """Return whether commands point exactly along shelf width (world +/-Y)."""

    angle_tensor = torch.as_tensor(angle)
    if not angle_tensor.dtype.is_floating_point:
        angle_tensor = angle_tensor.to(dtype=torch.get_default_dtype())
    left = torch.abs(circular_angle_difference(angle_tensor, SHELF_LEFT_ANGLE))
    right = torch.abs(circular_angle_difference(angle_tensor, SHELF_RIGHT_ANGLE))
    return (left <= LATERAL_ANGLE_TOLERANCE) | (right <= LATERAL_ANGLE_TOLERANCE)


def sample_sweep_angles(
    shape: int | Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the reference task's equiprobable left/right shelf directions."""

    if isinstance(shape, int):
        sample_shape = (shape,)
    else:
        sample_shape = tuple(shape)
    if any(size < 0 for size in sample_shape):
        raise ValueError("shape dimensions must be non-negative")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    side = torch.randint(
        0, 2, sample_shape, device=device, generator=generator, dtype=torch.int64
    )
    sign = 2.0 * side.to(dtype=dtype) - 1.0
    return sign * (math.pi / 2.0)


def direction_from_angle(angle_s: Tensor | float) -> Tensor:
    """Convert a permitted lateral angle to an exact world +/-Y direction."""

    angle_tensor = torch.as_tensor(angle_s)
    lateral_sign = torch.where(
        torch.sin(angle_tensor) >= 0.0,
        torch.ones_like(angle_tensor),
        -torch.ones_like(angle_tensor),
    )
    return torch.stack(
        (
            torch.zeros_like(angle_tensor),
            lateral_sign,
            torch.zeros_like(angle_tensor),
        ),
        dim=-1,
    )


def goal_from_command(
    initial_object_position_s: Tensor,
    angle_s: Tensor | float,
    distance: Tensor | float,
) -> Tensor:
    """Compute the fixed episode goal ``p_o,0 + L d`` in frame S."""

    if initial_object_position_s.ndim < 1 or initial_object_position_s.shape[-1] != 3:
        raise ValueError("initial_object_position_s must have shape (..., 3)")
    angle_tensor = torch.as_tensor(
        angle_s,
        dtype=initial_object_position_s.dtype,
        device=initial_object_position_s.device,
    )
    distance_tensor = torch.as_tensor(
        distance,
        dtype=initial_object_position_s.dtype,
        device=initial_object_position_s.device,
    )
    if torch.any(distance_tensor < 0):
        raise ValueError("command distance must be non-negative")
    return initial_object_position_s + distance_tensor.unsqueeze(-1) * direction_from_angle(
        angle_tensor
    )
