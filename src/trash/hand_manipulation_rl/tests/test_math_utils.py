from __future__ import annotations

import math

import torch

from hand_manipulation_rl.math_utils import (
    compose_c_frame_delta,
    matrix_to_rotation_vector,
    offset_point_jacobian,
    offset_point_velocity,
    oriented_box_overlap,
    relative_c_pose,
    rotation_vector_to_matrix,
    skew_symmetric,
    transform_wrench_f_to_c,
)


DTYPE = torch.float64


def test_skew_matrix_matches_cross_product() -> None:
    vector = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.2, 1.2]], dtype=DTYPE)
    other = torch.tensor([[4.0, -2.0, 0.5], [0.3, 0.1, -0.2]], dtype=DTYPE)
    actual = (skew_symmetric(vector) @ other.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(actual, torch.linalg.cross(vector, other, dim=-1))


def test_rotation_vector_round_trip_including_pi() -> None:
    rotation_vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.3], [math.pi, 0.0, 0.0]], dtype=DTYPE
    )
    matrices = rotation_vector_to_matrix(rotation_vectors)
    recovered = matrix_to_rotation_vector(matrices)
    assert torch.allclose(recovered, rotation_vectors, atol=1.0e-7)


def test_c_frame_delta_uses_current_local_frame() -> None:
    position = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    rotation = rotation_vector_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2], dtype=DTYPE))
    target_position, target_rotation = compose_c_frame_delta(
        position,
        rotation,
        torch.tensor([0.1, 0.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE),
    )
    assert torch.allclose(target_position, torch.tensor([1.0, 2.1, 3.0], dtype=DTYPE))
    assert torch.allclose(target_rotation, rotation)


def test_offset_velocity_and_jacobian_use_same_fixed_point() -> None:
    linear_velocity = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
    angular_velocity = torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE)
    offset = torch.tensor([0.0, 0.5, 0.0], dtype=DTYPE)
    assert torch.allclose(
        offset_point_velocity(linear_velocity, angular_velocity, offset),
        torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE),
    )

    linear_jacobian = torch.zeros(3, 3, dtype=DTYPE)
    angular_jacobian = torch.eye(3, dtype=DTYPE)
    expected = -skew_symmetric(offset)
    assert torch.allclose(
        offset_point_jacobian(linear_jacobian, angular_jacobian, offset), expected
    )


def test_relative_pose_is_expressed_in_c0() -> None:
    rotation_w_c0 = rotation_vector_to_matrix(
        torch.tensor([0.0, 0.0, math.pi / 2], dtype=DTYPE)
    )
    position, rotation_vector = relative_c_pose(
        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        rotation_w_c0,
        torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE),
        rotation_w_c0,
    )
    assert torch.allclose(position, torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE), atol=1e-8)
    assert torch.allclose(rotation_vector, torch.zeros(3, dtype=DTYPE), atol=1e-8)


def test_oriented_box_overlap_covers_face_edge_and_separation_axes() -> None:
    identity = torch.eye(3, dtype=DTYPE)
    center = torch.zeros(3, dtype=DTYPE)
    half = torch.tensor([1.0, 0.4, 0.25], dtype=DTYPE)
    quarter_turn = rotation_vector_to_matrix(
        torch.tensor([0.0, 0.0, math.pi / 4.0], dtype=DTYPE)
    )

    assert bool(oriented_box_overlap(center, identity, half, center, quarter_turn, half))
    assert bool(
        oriented_box_overlap(
            center,
            identity,
            half,
            torch.tensor([2.0, 0.0, 0.0], dtype=DTYPE),
            identity,
            half,
        )
    ), "Touching boxes are conservatively classified as overlapping"
    assert not bool(
        oriented_box_overlap(
            center,
            identity,
            half,
            torch.tensor([2.01, 0.0, 0.0], dtype=DTYPE),
            identity,
            half,
        )
    )


def test_oriented_box_overlap_broadcasts_over_environments_and_bodies() -> None:
    identity = torch.eye(3, dtype=DTYPE)
    centers_a = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
        dtype=DTYPE,
    )
    rotations_a = identity.expand(2, 2, 3, 3)
    half_a = torch.full((1, 2, 3), 0.5, dtype=DTYPE)
    centers_b = torch.zeros((2, 1, 3), dtype=DTYPE)
    rotations_b = identity.expand(2, 1, 3, 3)
    half_b = torch.full((1, 1, 3), 0.25, dtype=DTYPE)
    overlap = oriented_box_overlap(
        centers_a,
        rotations_a,
        half_a,
        centers_b,
        rotations_b,
        half_b,
    )
    assert torch.equal(overlap, torch.tensor([[True, False], [True, False]]))


def test_wrench_transform_shifts_moment_reference() -> None:
    force, moment = transform_wrench_f_to_c(
        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        torch.zeros(3, dtype=DTYPE),
        torch.eye(3, dtype=DTYPE),
        torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
    )
    assert torch.allclose(force, torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE))
    assert torch.allclose(moment, torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE))
