"""Reset FK correctness and simulator-I/O reduction contracts."""

import math
from pathlib import Path

import pytest
import torch
import numpy as np

from hand_manipulation_test.action_math import (
    quaternion_from_rotation_vector,
    quaternion_to_matrix,
)
from hand_manipulation_test.reset_kinematics import (
    SerialArmKinematics,
    NumpySerialArmKinematics,
    rotation_vector_from_quaternion,
    serial_arm_kinematics_from_usd,
)


def _model():
    dtype = torch.float64
    parent = torch.zeros(6, 7, dtype=dtype)
    parent[:, 3] = 1.
    parent[:, :3] = torch.tensor([[.1, 0., .2], [0., -.1, .1], [.2, 0., 0.],
                                  [.1, .1, 0.], [0., 0., .1], [.1, 0., 0.]])
    child = torch.zeros_like(parent)
    child[:, 3] = 1.
    child[:, :3] = torch.tensor([[.01, -.02, .03]]).expand(6, -1)
    child[:, 3:] = quaternion_from_rotation_vector(torch.tensor([[.1, -.05, .03]], dtype=dtype).expand(6, -1))
    axes = torch.eye(3, dtype=dtype).repeat(2, 1)
    control = torch.tensor([.01, .05, .1, 1., 0., 0., 0.], dtype=dtype)
    return SerialArmKinematics(parent, child, axes, control)


def _matrix(pose):
    value = torch.eye(4, dtype=pose.dtype).expand(pose.shape[:-1] + (4, 4)).clone()
    value[..., :3, :3] = quaternion_to_matrix(pose[..., 3:])
    value[..., :3, 3] = pose[..., :3]
    return value


def test_cached_fk_matches_independent_homogeneous_chain_and_geometric_jacobian():
    model = _model()
    q = torch.tensor([[.2, -.7, .5, .3, -.4, .9], [-.6, .3, -.2, 1.2, .4, -.8]], dtype=torch.float64)
    root = torch.tensor([[0., 0., .79505, 0., 0., 0., 1.], [2.5, -5., .79505, 0., 0., 0., 1.]], dtype=q.dtype)
    transform = _matrix(root)
    origins, axes_w = [], []
    for index in range(6):
        transform = transform @ _matrix(model.parent_frames[index])
        origins.append(transform[:, :3, 3].clone())
        axes_w.append((transform[:, :3, :3] @ model.axes[index, :, None]).squeeze(-1))
        rotation = torch.zeros(2, 7, dtype=q.dtype)
        rotation[:, 3:] = quaternion_from_rotation_vector(q[:, index, None] * model.axes[index])
        transform = transform @ _matrix(rotation) @ _matrix(model.child_inverse_frames[index])
    transform = transform @ _matrix(model.control_frame)
    position, quaternion, numerical = model.pose_and_jacobian(q, root, epsilon=1.e-6)
    torch.testing.assert_close(position, transform[:, :3, 3], atol=1.e-12, rtol=1.e-12)
    torch.testing.assert_close(quaternion_to_matrix(quaternion), transform[:, :3, :3], atol=1.e-12, rtol=1.e-12)
    expected = torch.stack([torch.cat((torch.linalg.cross(axis, position - origin), axis), -1)
                            for origin, axis in zip(origins, axes_w)], -1)
    torch.testing.assert_close(numerical, expected, atol=4.e-7, rtol=4.e-6)


def test_six_jacobian_columns_are_one_batch_and_do_not_mutate_joint_or_root_state():
    model = _model()
    q = torch.linspace(-.8, .8, 24, dtype=torch.float64).reshape(4, 6)
    root = torch.zeros(4, 7, dtype=q.dtype)
    root[:, 3] = 1.
    saved_q, saved_root = q.clone(), root.clone()
    shapes = []
    pose = model.pose
    def counted_pose(positions, roots):
        shapes.append(tuple(positions.shape))
        return pose(positions, roots)
    model.pose = counted_pose
    model.pose_and_jacobian(q, root)
    assert shapes == [(4, 7, 6)]
    torch.testing.assert_close(q, saved_q)
    torch.testing.assert_close(root, saved_root)


def test_rotation_vector_has_finite_zero_and_shortest_quaternion_sign():
    vector = torch.tensor([[0., 0., 0.], [1.e-9, 0., 0.], [.3, -.2, .1]], dtype=torch.float64)
    quaternion = quaternion_from_rotation_vector(vector)
    torch.testing.assert_close(rotation_vector_from_quaternion(quaternion), vector)
    torch.testing.assert_close(rotation_vector_from_quaternion(-quaternion), vector)
    assert bool(torch.isfinite(rotation_vector_from_quaternion(quaternion)).all())


def test_numpy_batch_matches_device_math_at_every_jacobian_column():
    model = _model()
    cpu_model = NumpySerialArmKinematics(model)
    q = torch.linspace(-1., 1., 48, dtype=torch.float64).reshape(8, 6)
    root = torch.zeros(8, 7, dtype=q.dtype)
    root[:, 3] = 1.
    root[:, :3] = torch.linspace(-5., 5., 24).reshape(8, 3)
    expected = model.pose_and_jacobian(q, root)
    actual = cpu_model.pose_and_jacobian(q.numpy(), root.numpy())
    for tensor, array in zip(expected, actual):
        torch.testing.assert_close(torch.from_numpy(array), tensor, atol=2.e-11, rtol=2.e-11)


def test_fixed_xyz_cross_is_exact_for_broadcasting_frames_and_joint_batches():
    left = np.arange(168, dtype=np.float64).reshape(8, 7, 3) / 7. - 12.
    for right in (np.array([.1, -.2, .3]), left[:, :1] + .03, left * .17 - .09):
        np.testing.assert_array_equal(NumpySerialArmKinematics.cross_xyz(left, right), np.cross(left, right))


def test_cpu_ik_preserves_reached_rows_and_joint_limits_without_scene_io():
    model = _model()
    cpu_model = NumpySerialArmKinematics(model)
    target_q = torch.tensor([[.2, -.7, .5, .3, -.4, .9], [-.6, .3, -.2, 1.2, .4, -.8]], dtype=torch.float64)
    root = torch.zeros(2, 7, dtype=target_q.dtype)
    root[:, 3] = 1.
    p, r = model.pose(target_q, root)
    target = torch.cat((p, r), -1).numpy()
    seed = target_q.numpy().copy()
    seed[1] += np.array([.12, -.1, .08, -.1, .1, -.12])
    saved_seed = seed.copy()
    lower, upper = np.full((2, 6), -2.), np.full((2, 6), 2.)
    q, iterations, batches = cpu_model.solve(seed, root.numpy(), target, lower, upper,
        max_iterations=80, damping=.045, step_size=.65, position_error_step=.06,
        orientation_error_step=.25, delta_limit=.18, position_tolerance=.003, orientation_tolerance=.05)
    np.testing.assert_array_equal(q[0], saved_seed[0])
    np.testing.assert_array_equal(seed, saved_seed)
    assert iterations[0] == 1 and 1 < iterations[1] <= 80 and batches == iterations.max()
    assert np.all((q >= lower) & (q <= upper))
    p, r = cpu_model.pose(q, root.numpy())
    assert np.all(np.linalg.norm(p - target[:, :3], axis=-1) <= .003)
    inverse = r * np.array([1., -1., -1., -1.])
    error = cpu_model.rotation_vector(cpu_model.multiply(target[:, 3:], inverse))
    assert np.all(np.linalg.norm(error, axis=-1) <= .05)


def test_usd_joint_frames_match_source_zero_pose_and_env_translation():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom
    asset = Path(__file__).parents[1] / "hand_manipulation_test/assets/data/ur5e_inspire_usd/ur5e_inspire.usd"
    stage = Usd.Stage.Open(str(asset))
    root_path = str(stage.GetDefaultPrim().GetPath())
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    model = serial_arm_kinematics_from_usd(stage, root_path, names, "inspire_base_link", (0., .05, .1),
                                         (1., 0., 0., 0.), device="cpu", dtype=torch.float64)
    cache = UsdGeom.XformCache()
    base = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path + "/base_link")))
    q = base.GetRotation().GetQuat()
    root = torch.tensor([[*base.GetTranslation(), q.GetReal(), *q.GetImaginary()]], dtype=torch.float64)
    position, _ = model.pose(torch.zeros(1, 6, dtype=torch.float64), root)
    expected = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path + "/inspire_base_link")).Transform(Gf.Vec3d(0., .05, .1))
    # USD local joint frames are single-precision; the composed mesh transform
    # is double-precision. Their zero-pose mismatch stays well below reset tol.
    torch.testing.assert_close(position[0], torch.tensor(list(expected)), atol=2.e-6, rtol=0., check_dtype=False)
    shifted = root.clone()
    shifted[:, :3] += torch.tensor([[2.5, -5., 0.]])
    moved, _ = model.pose(torch.zeros(1, 6, dtype=torch.float64), shifted)
    torch.testing.assert_close(moved - position, torch.tensor([[2.5, -5., 0.]], dtype=root.dtype))
