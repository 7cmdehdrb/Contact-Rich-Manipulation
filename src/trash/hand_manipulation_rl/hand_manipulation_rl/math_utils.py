"""Batched rigid-body math used by the blind-sweeping environment.

This module deliberately depends only on PyTorch.  Rotations follow the
column-vector convention: ``rotation_ab`` maps a vector expressed in frame B
to frame A.
"""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def _require_vector3(value: Tensor, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3); got {tuple(value.shape)}")


def _require_matrix3(value: Tensor, name: str) -> None:
    if value.ndim < 2 or value.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have shape (..., 3, 3); got {tuple(value.shape)}")


def skew_symmetric(vector: Tensor) -> Tensor:
    """Return the skew matrix ``[vector]_x`` for one or more 3D vectors."""

    _require_vector3(vector, "vector")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    values = (zero, -z, y, z, zero, -x, -y, x, zero)
    return torch.stack(values, dim=-1).reshape(vector.shape[:-1] + (3, 3))


def rotation_vector_to_matrix(rotation_vector: Tensor) -> Tensor:
    """Convert an axis-angle rotation vector to a rotation matrix.

    A Taylor expansion is used near zero so a zero action produces an exact,
    finite identity transform.
    """

    _require_vector3(rotation_vector, "rotation_vector")
    theta_sq = torch.sum(rotation_vector * rotation_vector, dim=-1, keepdim=True)
    eps = torch.finfo(rotation_vector.dtype).eps
    theta = torch.sqrt(torch.clamp_min(theta_sq, eps * eps))

    theta_fourth = theta_sq * theta_sq
    a_small = 1.0 - theta_sq / 6.0 + theta_fourth / 120.0
    b_small = 0.5 - theta_sq / 24.0 + theta_fourth / 720.0
    a = torch.where(theta_sq > eps * eps, torch.sin(theta) / theta, a_small)
    b = torch.where(
        theta_sq > eps * eps,
        (1.0 - torch.cos(theta)) / torch.clamp_min(theta_sq, eps * eps),
        b_small,
    )

    skew = skew_symmetric(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(rotation_vector.shape[:-1] + (3, 3))
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def matrix_to_rotation_vector(rotation_matrix: Tensor) -> Tensor:
    """Convert a rotation matrix to its principal axis-angle vector.

    The returned angle is in ``[0, pi]``.  Dedicated branches keep both the
    identity and 180-degree cases finite.
    """

    _require_matrix3(rotation_matrix, "rotation_matrix")
    dtype = rotation_matrix.dtype
    eps = torch.finfo(dtype).eps
    trace = torch.diagonal(rotation_matrix, dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(cosine)

    vee = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )

    sin_angle = torch.sin(angle)
    safe_sin = torch.clamp(torch.abs(sin_angle), min=eps)
    regular = vee * (angle / (2.0 * safe_sin)).unsqueeze(-1)
    small = 0.5 * vee

    # Near pi, the skew part vanishes.  Recover the axis from R + I, choosing
    # the largest diagonal element to avoid division by a small component.
    diagonal = torch.diagonal(rotation_matrix, dim1=-2, dim2=-1)
    largest = torch.argmax(diagonal, dim=-1)
    minimum_axis = torch.sqrt(torch.as_tensor(eps, dtype=dtype, device=rotation_matrix.device))

    x = torch.sqrt(torch.clamp_min((rotation_matrix[..., 0, 0] + 1.0) * 0.5, 0.0))
    x_safe = torch.clamp_min(x, minimum_axis)
    axis_x = torch.stack(
        (
            x,
            (rotation_matrix[..., 0, 1] + rotation_matrix[..., 1, 0]) / (4.0 * x_safe),
            (rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0]) / (4.0 * x_safe),
        ),
        dim=-1,
    )

    y = torch.sqrt(torch.clamp_min((rotation_matrix[..., 1, 1] + 1.0) * 0.5, 0.0))
    y_safe = torch.clamp_min(y, minimum_axis)
    axis_y = torch.stack(
        (
            (rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1]) / (4.0 * y_safe),
            y,
            (rotation_matrix[..., 1, 2] + rotation_matrix[..., 2, 1]) / (4.0 * y_safe),
        ),
        dim=-1,
    )

    z = torch.sqrt(torch.clamp_min((rotation_matrix[..., 2, 2] + 1.0) * 0.5, 0.0))
    z_safe = torch.clamp_min(z, minimum_axis)
    axis_z = torch.stack(
        (
            (rotation_matrix[..., 2, 0] + rotation_matrix[..., 0, 2]) / (4.0 * z_safe),
            (rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2]) / (4.0 * z_safe),
            z,
        ),
        dim=-1,
    )

    axis_pi = torch.where((largest == 0).unsqueeze(-1), axis_x, axis_y)
    axis_pi = torch.where((largest == 2).unsqueeze(-1), axis_z, axis_pi)
    axis_pi = axis_pi / torch.clamp_min(torch.linalg.vector_norm(axis_pi, dim=-1, keepdim=True), eps)
    near_pi = axis_pi * angle.unsqueeze(-1)

    small_mask = angle < 1.0e-5
    pi_mask = (torch.pi - angle) < 1.0e-4
    result = torch.where(small_mask.unsqueeze(-1), small, regular)
    return torch.where(pi_mask.unsqueeze(-1), near_pi, result)


def compose_c_frame_delta(
    position_b_c: Tensor,
    rotation_b_c: Tensor,
    delta_position_c: Tensor,
    delta_rotation_c: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a local Cartesian increment to the current actual C pose.

    The target is recomputed from the supplied current pose; this function does
    not accumulate an earlier desired pose.
    """

    _require_vector3(position_b_c, "position_b_c")
    _require_matrix3(rotation_b_c, "rotation_b_c")
    _require_vector3(delta_position_c, "delta_position_c")
    _require_vector3(delta_rotation_c, "delta_rotation_c")
    delta_position_b = (rotation_b_c @ delta_position_c.unsqueeze(-1)).squeeze(-1)
    target_position_b = position_b_c + delta_position_b
    target_rotation_b_c = rotation_b_c @ rotation_vector_to_matrix(delta_rotation_c)
    return target_position_b, target_rotation_b_c


def offset_point_velocity(
    linear_velocity_b_h: Tensor,
    angular_velocity_b_h: Tensor,
    offset_h_c_b: Tensor,
) -> Tensor:
    """Shift a rigid body's linear velocity from H's origin to point C."""

    _require_vector3(linear_velocity_b_h, "linear_velocity_b_h")
    _require_vector3(angular_velocity_b_h, "angular_velocity_b_h")
    _require_vector3(offset_h_c_b, "offset_h_c_b")
    return linear_velocity_b_h + torch.linalg.cross(
        angular_velocity_b_h, offset_h_c_b, dim=-1
    )


def offset_point_jacobian(
    linear_jacobian_b_h: Tensor,
    angular_jacobian_b_h: Tensor,
    offset_h_c_b: Tensor,
) -> Tensor:
    """Shift the translational Jacobian from H's origin to fixed point C."""

    if linear_jacobian_b_h.ndim < 2 or linear_jacobian_b_h.shape[-2] != 3:
        raise ValueError("linear_jacobian_b_h must have shape (..., 3, dof)")
    if angular_jacobian_b_h.shape != linear_jacobian_b_h.shape:
        raise ValueError("angular and linear Jacobians must have identical shapes")
    _require_vector3(offset_h_c_b, "offset_h_c_b")
    return linear_jacobian_b_h - skew_symmetric(offset_h_c_b) @ angular_jacobian_b_h


def relative_c_pose(
    position_w_c0: Tensor,
    rotation_w_c0: Tensor,
    position_w_ct: Tensor,
    rotation_w_ct: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return C_t relative to C_0 as position and a rotation vector."""

    _require_vector3(position_w_c0, "position_w_c0")
    _require_matrix3(rotation_w_c0, "rotation_w_c0")
    _require_vector3(position_w_ct, "position_w_ct")
    _require_matrix3(rotation_w_ct, "rotation_w_ct")
    rotation_c0_w = rotation_w_c0.transpose(-1, -2)
    relative_position = (
        rotation_c0_w @ (position_w_ct - position_w_c0).unsqueeze(-1)
    ).squeeze(-1)
    relative_rotation = rotation_c0_w @ rotation_w_ct
    return relative_position, matrix_to_rotation_vector(relative_rotation)


def oriented_box_overlap(
    center_w_a: Tensor,
    rotation_w_a: Tensor,
    half_extent_a: Tensor,
    center_w_b: Tensor,
    rotation_w_b: Tensor,
    half_extent_b: Tensor,
) -> Tensor:
    """Return whether two 3-D oriented boxes overlap or touch.

    The implementation is the complete 15-axis separating-axis test: three
    face normals from each box and the nine pairwise edge cross-products.
    Inputs may have any broadcast-compatible leading dimensions.  Treating
    touching boxes as overlapping makes this suitable for conservative reset
    collision certification.
    """

    _require_vector3(center_w_a, "center_w_a")
    _require_vector3(center_w_b, "center_w_b")
    _require_matrix3(rotation_w_a, "rotation_w_a")
    _require_matrix3(rotation_w_b, "rotation_w_b")
    _require_vector3(half_extent_a, "half_extent_a")
    _require_vector3(half_extent_b, "half_extent_b")
    if torch.any(half_extent_a < 0.0) or torch.any(half_extent_b < 0.0):
        raise ValueError("Oriented-box half extents must be non-negative")

    rotation_a_b = rotation_w_a.transpose(-1, -2) @ rotation_w_b
    translation_a = (
        rotation_w_a.transpose(-1, -2)
        @ (center_w_b - center_w_a).unsqueeze(-1)
    ).squeeze(-1)

    # The epsilon handles nearly parallel axes.  Adding it to |R| slightly
    # enlarges the projected boxes, preserving the conservative direction.
    epsilon = 8.0 * torch.finfo(rotation_a_b.dtype).eps
    abs_rotation = torch.abs(rotation_a_b) + epsilon

    projection_b_on_a = (
        abs_rotation * half_extent_b.unsqueeze(-2)
    ).sum(dim=-1)
    separated = torch.any(
        torch.abs(translation_a) > half_extent_a + projection_b_on_a,
        dim=-1,
    )

    translation_b = (
        translation_a.unsqueeze(-2) @ rotation_a_b
    ).squeeze(-2)
    projection_a_on_b = (
        abs_rotation * half_extent_a.unsqueeze(-1)
    ).sum(dim=-2)
    separated |= torch.any(
        torch.abs(translation_b) > half_extent_b + projection_a_on_b,
        dim=-1,
    )

    for axis_a in range(3):
        next_a = (axis_a + 1) % 3
        last_a = (axis_a + 2) % 3
        for axis_b in range(3):
            next_b = (axis_b + 1) % 3
            last_b = (axis_b + 2) % 3
            distance = torch.abs(
                translation_a[..., last_a] * rotation_a_b[..., next_a, axis_b]
                - translation_a[..., next_a] * rotation_a_b[..., last_a, axis_b]
            )
            radius_a = (
                half_extent_a[..., next_a] * abs_rotation[..., last_a, axis_b]
                + half_extent_a[..., last_a] * abs_rotation[..., next_a, axis_b]
            )
            radius_b = (
                half_extent_b[..., next_b] * abs_rotation[..., axis_a, last_b]
                + half_extent_b[..., last_b] * abs_rotation[..., axis_a, next_b]
            )
            separated |= distance > radius_a + radius_b

    return ~separated


def transform_wrench_f_to_c(
    force_f: Tensor,
    moment_at_f_f: Tensor,
    rotation_c_f: Tensor,
    offset_c_f_c: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate a wrench from F to C and shift its moment reference to C.

    ``offset_c_f_c`` is the vector from C to F, expressed in C.
    """

    _require_vector3(force_f, "force_f")
    _require_vector3(moment_at_f_f, "moment_at_f_f")
    _require_matrix3(rotation_c_f, "rotation_c_f")
    _require_vector3(offset_c_f_c, "offset_c_f_c")
    force_c = (rotation_c_f @ force_f.unsqueeze(-1)).squeeze(-1)
    rotated_moment_c = (rotation_c_f @ moment_at_f_f.unsqueeze(-1)).squeeze(-1)
    moment_at_c_c = rotated_moment_c + torch.linalg.cross(
        offset_c_f_c, force_c, dim=-1
    )
    return force_c, moment_at_c_c
