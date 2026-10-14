"""Cached, simulator-independent serial-arm FK for reset-only batched IK.

Joint frames are supplied from the composed USD once. No simulator state is
written while evaluating the six finite-difference Jacobian columns.
"""

from __future__ import annotations

import torch
import numpy as np

from .action_math import quaternion_conjugate, quaternion_multiply, quaternion_rotate


def cached_ik_convergence_tolerances(position_tolerance: float, orientation_tolerance: float):
    """Leave room for float32 joint storage and the independent PhysX FK gate.

    These are internal stopping criteria. The configured final tolerances and
    collision clearance remain authoritative and are never relaxed.
    """

    return .95 * position_tolerance, .95 * orientation_tolerance


def bounded_dls_correction(jacobian, position_error, rotation_error, *, damping, step_size,
                           position_error_step, orientation_error_step, delta_limit):
    """One measured-FK correction with the existing reset's step bounds."""

    position_error = position_error * (position_error_step / position_error.norm(dim=-1, keepdim=True).clamp_min(1.e-9)).clamp(max=1.)
    rotation_error = rotation_error * (orientation_error_step / rotation_error.norm(dim=-1, keepdim=True).clamp_min(1.e-9)).clamp(max=1.)
    error = torch.cat((position_error, rotation_error), dim=-1).unsqueeze(-1)
    identity = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
    transpose = jacobian.transpose(1, 2)
    solution = torch.linalg.solve(jacobian @ transpose + damping**2 * identity, error)
    return (step_size * (transpose @ solution).squeeze(-1)).clamp(-delta_limit, delta_limit)


def rotation_vector_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation vector, using Isaac Lab's small-angle rule."""

    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    imaginary = quaternion[..., 1:]
    sine = torch.linalg.vector_norm(imaginary, dim=-1)
    half_angle = torch.atan2(sine, quaternion[..., 0])
    angle = 2.0 * half_angle
    ratio = torch.where(angle.abs() > 1.0e-6, sine / angle.clamp_min(1.0e-12), .5 - angle.square() / 48.0)
    return imaginary / ratio.clamp_min(1.0e-12).unsqueeze(-1)


class SerialArmKinematics:
    """Compose six revolute joint frames and the fixed wrist-to-control pose.

    ``parent_frames`` maps joint frame to parent body; ``child_inverse_frames``
    maps child body to joint frame. Poses use xyz followed by a wxyz quaternion.
    All tensors remain on the selected device, including perturbation batches.
    """

    def __init__(self, parent_frames, child_inverse_frames, axes, control_frame):
        self.parent_frames = parent_frames
        self.child_inverse_frames = child_inverse_frames
        self.axes = axes
        self.control_frame = control_frame
        self._perturbations = torch.cat((torch.zeros(1, 6, device=axes.device, dtype=axes.dtype),
                                         torch.eye(6, device=axes.device, dtype=axes.dtype)), dim=0)
        if parent_frames.shape != (6, 7) or child_inverse_frames.shape != (6, 7) or axes.shape != (6, 3):
            raise ValueError("A six-joint serial chain requires six poses and six axes")
        if control_frame.shape != (7,):
            raise ValueError("The fixed control frame must be an xyz/wxyz pose")

    @staticmethod
    def _compose(position, quaternion, frame):
        frame = frame.expand(quaternion.shape[:-1] + (7,))
        position = position + quaternion_rotate(quaternion, frame[..., :3])
        quaternion = quaternion_multiply(quaternion, frame[..., 3:])
        return position, quaternion

    def pose(self, joint_positions: torch.Tensor, root_pose: torch.Tensor):
        position, quaternion = root_pose[..., :3], root_pose[..., 3:]
        for index in range(6):
            position, quaternion = self._compose(position, quaternion, self.parent_frames[index])
            half_angle = .5 * joint_positions[..., index:index + 1]
            rotation = torch.cat((half_angle.cos(), half_angle.sin() * self.axes[index]), dim=-1)
            quaternion = quaternion_multiply(quaternion, rotation)
            position, quaternion = self._compose(position, quaternion, self.child_inverse_frames[index])
        position, quaternion = self._compose(position, quaternion, self.control_frame)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return position, quaternion

    def pose_and_jacobian(self, joint_positions: torch.Tensor, root_pose: torch.Tensor, epsilon: float = 1.0e-3):
        """Evaluate the nominal pose and all six columns in one FK batch."""

        perturbed = joint_positions[:, None, :] + epsilon * self._perturbations
        positions, quaternions = self.pose(perturbed, root_pose[:, None, :])
        position, quaternion = positions[:, 0], quaternions[:, 0]
        translation = (positions[:, 1:] - position[:, None]) / epsilon
        inverse = quaternion_conjugate(quaternion[:, None])
        rotation = rotation_vector_from_quaternion(quaternion_multiply(quaternions[:, 1:], inverse)) / epsilon
        jacobian = torch.cat((translation, rotation), dim=-1).transpose(-1, -2)
        return position, quaternion, jacobian


class NumpySerialArmKinematics:
    """Small reset batches with no device transfer inside the bounded IK loop."""

    def __init__(self, model: SerialArmKinematics):
        self.parent_frames = model.parent_frames.detach().cpu().numpy().astype(np.float64)
        self.child_frames = model.child_inverse_frames.detach().cpu().numpy().astype(np.float64)
        self.axes = model.axes.detach().cpu().numpy().astype(np.float64)
        self.control_frame = model.control_frame.detach().cpu().numpy().astype(np.float64)
        self.perturbations = np.concatenate((np.zeros((1, 6)), np.eye(6)), axis=0)

    @staticmethod
    def cross_xyz(left, right):
        """Fixed xyz cross product without NumPy's generic axis machinery."""

        return np.stack((left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
                         left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
                         left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]), axis=-1)

    @staticmethod
    def multiply(left, right):
        lw, lv, rw, rv = left[..., :1], left[..., 1:], right[..., :1], right[..., 1:]
        return np.concatenate((lw * rw - (lv * rv).sum(-1, keepdims=True),
                               lw * rv + rw * lv + NumpySerialArmKinematics.cross_xyz(lv, rv)), -1)

    @staticmethod
    def rotate(quaternion, vector):
        quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
        qv = quaternion[..., 1:]
        uv = 2.0 * NumpySerialArmKinematics.cross_xyz(qv, vector)
        return vector + quaternion[..., :1] * uv + NumpySerialArmKinematics.cross_xyz(qv, uv)

    @classmethod
    def compose(cls, position, quaternion, frame):
        return position + cls.rotate(quaternion, frame[:3]), cls.multiply(quaternion, frame[3:])

    @staticmethod
    def rotation_vector(quaternion):
        quaternion = np.where(quaternion[..., :1] < 0., -quaternion, quaternion)
        imaginary = quaternion[..., 1:]
        half_angle = np.arctan2(np.linalg.norm(imaginary, axis=-1), quaternion[..., 0])
        angle = 2.0 * half_angle
        ratio = np.where(np.abs(angle) > 1.e-6, np.sin(half_angle) / np.maximum(angle, 1.e-12), .5 - angle**2 / 48.)
        return imaginary / ratio[..., None]

    def pose(self, joint_positions, root_pose):
        position, quaternion = root_pose[..., :3], root_pose[..., 3:]
        for index in range(6):
            position, quaternion = self.compose(position, quaternion, self.parent_frames[index])
            half_angle = .5 * joint_positions[..., index:index + 1]
            rotation = np.concatenate((np.cos(half_angle), np.sin(half_angle) * self.axes[index]), axis=-1)
            quaternion = self.multiply(quaternion, rotation)
            position, quaternion = self.compose(position, quaternion, self.child_frames[index])
        position, quaternion = self.compose(position, quaternion, self.control_frame)
        return position, quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)

    def pose_and_jacobian(self, joint_positions, root_pose, epsilon=1.e-3):
        positions, quaternions = self.pose(joint_positions[:, None] + epsilon * self.perturbations, root_pose[:, None])
        position, quaternion = positions[:, 0], quaternions[:, 0]
        inverse = quaternion[:, None] * np.array([1., -1., -1., -1.])
        rotation = self.rotation_vector(self.multiply(quaternions[:, 1:], inverse)) / epsilon
        translation = (positions[:, 1:] - position[:, None]) / epsilon
        return position, quaternion, np.concatenate((translation, rotation), axis=-1).transpose(0, 2, 1)

    def solve(self, positions, root, target, lower, upper, *, max_iterations, damping, step_size,
              position_error_step, orientation_error_step, delta_limit, position_tolerance, orientation_tolerance):
        """Match the existing pending-row finite-difference DLS solve on CPU."""

        q = positions.astype(np.float64, copy=True)
        counts = np.zeros(len(q), dtype=np.int64)
        converged = np.zeros(len(q), dtype=bool)
        batches = 0
        for _ in range(max_iterations):
            rows = np.flatnonzero(~converged)
            if not len(rows):
                break
            counts[rows] += 1
            batches += 1
            position, quaternion, jacobian = self.pose_and_jacobian(q[rows], root[rows])
            pe = target[rows, :3] - position
            inverse = quaternion * np.array([1., -1., -1., -1.])
            re = self.rotation_vector(self.multiply(target[rows, 3:], inverse))
            reached = (np.linalg.norm(pe, axis=-1) <= position_tolerance) & (np.linalg.norm(re, axis=-1) <= orientation_tolerance)
            converged[rows[reached]] = True
            pending = rows[~reached]
            if not len(pending):
                break
            pe, re, jacobian = pe[~reached], re[~reached], jacobian[~reached]
            pe *= np.minimum(1., position_error_step / np.maximum(np.linalg.norm(pe, axis=-1, keepdims=True), 1.e-9))
            re *= np.minimum(1., orientation_error_step / np.maximum(np.linalg.norm(re, axis=-1, keepdims=True), 1.e-9))
            error = np.concatenate((pe, re), axis=-1)[..., None]
            transpose = jacobian.transpose(0, 2, 1)
            dls = np.linalg.solve(jacobian @ transpose + damping**2 * np.eye(6), error)
            delta = np.clip(step_size * (transpose @ dls)[..., 0], -delta_limit, delta_limit)
            q[pending] = np.clip(q[pending] + delta, lower[pending], upper[pending])
        return q, counts, batches


def serial_arm_kinematics_from_usd(stage, robot_root_path, joint_names, hand_body_name, control_position,
                                   control_quaternion, *, device, dtype=torch.float32):
    """Read the actual composed joint frames once, including the F/T mount."""

    from pxr import Gf, Usd, UsdPhysics

    def matrix(position, quaternion):
        value = Gf.Matrix4d(1.0)
        value.SetRotate(Gf.Quatd(quaternion))
        value.SetTranslateOnly(Gf.Vec3d(position))
        return value

    def pose(value):
        transform = Gf.Transform(value)
        q = transform.GetRotation().GetQuat().GetNormalized()
        return [*map(float, transform.GetTranslation()), float(q.GetReal()), *map(float, q.GetImaginary())]

    def frames(prim):
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            raise ValueError(f"Reset FK requires two body references for {prim.GetPath()}")
        parent = matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        child_inverse = matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()).GetInverse()
        return str(parents[0]), str(children[0]), parent, child_inverse

    parent_frames, child_frames, axes = [], [], []
    parent_body = robot_root_path + "/base_link"
    for name in joint_names:
        prim = stage.GetPrimAtPath(robot_root_path + "/joints/" + name)
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Reset FK requires the expected revolute joint {prim.GetPath()}")
        parent, child, local_parent, inverse_child = frames(prim)
        if parent != parent_body:
            raise ValueError(f"Reset FK serial chain mismatch at {name}: {parent} != {parent_body}")
        parent_frames.append(pose(local_parent))
        child_frames.append(pose(inverse_child))
        axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
        axes.append({"X": (1., 0., 0.), "Y": (0., 1., 0.), "Z": (0., 0., 1.)}[axis])
        parent_body = child
    fixed = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root_path)):
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(prim)
            if joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets():
                parent, child, local_parent, inverse_child = frames(prim)
                fixed.setdefault(parent, []).append((child, inverse_child * local_parent))
    target_body = robot_root_path + "/" + hand_body_name
    pending = [(parent_body, Gf.Matrix4d(1.0))]
    visited = set()
    hand_frame = None
    while pending:
        body, value = pending.pop()
        if body == target_body:
            hand_frame = value
            break
        if body in visited:
            continue
        visited.add(body)
        pending.extend((child, relative * value) for child, relative in fixed.get(body, ()))
    if hand_frame is None:
        raise ValueError(f"Reset FK cannot find the fixed mount from wrist to {target_body}")
    control = matrix(control_position, Gf.Quatd(control_quaternion[0], Gf.Vec3d(*control_quaternion[1:])))
    tensor = lambda value: torch.tensor(value, device=device, dtype=dtype)
    return SerialArmKinematics(tensor(parent_frames), tensor(child_frames), tensor(axes), tensor(pose(control * hand_frame)))


__all__ = ["SerialArmKinematics", "NumpySerialArmKinematics", "rotation_vector_from_quaternion", "serial_arm_kinematics_from_usd",
           "cached_ik_convergence_tolerances", "bounded_dls_correction"]
