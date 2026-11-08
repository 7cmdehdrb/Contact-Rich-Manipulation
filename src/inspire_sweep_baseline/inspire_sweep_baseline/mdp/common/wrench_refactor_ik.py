"""Batched reset IK from the spawned USD joint frames, without physics steps."""

import math
import torch
from isaaclab.utils.math import (
    quat_apply, quat_conjugate, quat_mul, axis_angle_from_quat,
)


def sample_initial_eef(target_local, sweep_local, default_local, generator,
                       front_clearance=(0.10, 0.13), lateral_offset=(0.10, 0.14),
                       height_jitter=0.01):
    """Outside the shelf front (+X), on the opposite Y side of the target.

    Pure push/pull (delta Y == 0) belongs to left, hence starts on +Y.
    Rotation is deliberately not sampled. Preserve the parent's default Z.
    """
    for name, bounds in (('front_clearance', front_clearance), ('lateral_offset', lateral_offset)):
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f'{name} must be an ordered positive interval')
    if not math.isfinite(height_jitter) or height_jitter < 0:
        raise ValueError('height_jitter must be finite and nonnegative')
    draw = torch.rand(target_local.shape, device=target_local.device,
                      dtype=target_local.dtype, generator=generator)
    side = torch.where(sweep_local[:, 1] <= 0, 1., -1.)
    result = default_local.clone()
    result[:, 0] = .20 + front_clearance[0] + draw[:, 0] * (front_clearance[1] - front_clearance[0])
    result[:, 1] = target_local[:, 1] + side * (
        lateral_offset[0] + draw[:, 1] * (lateral_offset[1] - lateral_offset[0]))
    result[:, 2] += (2 * draw[:, 2] - 1) * height_jitter
    return result


def initial_pose_quality(chain, angles, limits, max_condition=80.0, joint_margin=0.05):
    """Reject near-singular or joint-limit-adjacent reset poses."""
    _, _, jacobian = chain.forward(angles, jacobian=True)
    jacobian = jacobian.clone()
    jacobian[:, :3] /= 0.3
    finite = torch.isfinite(jacobian).all(dim=(-2, -1))
    singular = torch.linalg.svdvals(torch.nan_to_num(jacobian))
    condition = torch.where(singular[:, -1] > 1e-8,
                            singular[:, 0] / singular[:, -1].clamp_min(1e-12), torch.inf)
    margin = torch.minimum(angles - limits[..., 0], limits[..., 1] - angles).amin(-1)
    valid = finite & torch.isfinite(margin) & (condition <= max_condition) & (margin >= joint_margin)
    return valid, condition, margin


class ResetArmIK:
    """Six-axis serial chain using USD's parent/child joint-local transforms.

    FK and geometric Jacobians are evaluated in the base_link frame. Only the
    final converged solution is written to PhysX; other environments never step.
    """

    def __init__(self, joints, offset_pos, offset_quat, device):
        self.device = device
        self.joints = []
        for p0, q0, p1, q1, axis, column in joints:
            values = [torch.tensor(v, device=device, dtype=torch.float32) for v in (p0, q0, p1, q1)]
            values[1] = values[1] / values[1].norm()
            values[3] = values[3] / values[3].norm()
            self.joints.append((*values, axis, column))
        self.offset_pos = torch.tensor(offset_pos, device=device, dtype=torch.float32)
        self.offset_quat = torch.tensor(offset_quat, device=device, dtype=torch.float32)
        self.offset_quat /= self.offset_quat.norm()

    @classmethod
    def from_stage(cls, stage, robot_path, joint_names, offset_pos, offset_quat, device):
        from pxr import Usd, UsdPhysics

        incoming = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                children = joint.GetBody1Rel().GetTargets()
                parents = joint.GetBody0Rel().GetTargets()
                if len(children) == len(parents) == 1:
                    incoming[str(children[0])] = (prim, joint, str(parents[0]))
        current, root = robot_path + '/axia80_link', robot_path + '/base_link'
        chain, seen = [], set()
        def quaternion(value):
            return (float(value.GetReal()), *map(float, value.GetImaginary()))
        while current != root:
            if current in seen or current not in incoming:
                raise RuntimeError(f'Cannot trace reset IK chain at {current}')
            seen.add(current)
            prim, joint, parent = incoming[current]
            name = prim.GetName()
            column = joint_names.index(name) if name in joint_names else None
            if column is None and not prim.IsA(UsdPhysics.FixedJoint):
                raise RuntimeError(f'Unexpected movable joint in reset IK: {name}')
            axis = 'XYZ'.index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if column is not None else None
            chain.append((tuple(joint.GetLocalPos0Attr().Get()), quaternion(joint.GetLocalRot0Attr().Get()),
                          tuple(joint.GetLocalPos1Attr().Get()), quaternion(joint.GetLocalRot1Attr().Get()),
                          axis, column))
            current = parent
        chain.reverse()
        if sorted(j[-1] for j in chain if j[-1] is not None) != list(range(6)):
            raise RuntimeError('Reset IK requires exactly the six configured arm joints')
        return cls(chain, offset_pos, offset_quat, device)

    def forward(self, angles, jacobian=False):
        count = len(angles)
        pos = angles.new_zeros(count, 3)
        quat = angles.new_zeros(count, 4)
        quat[:, 0] = 1
        origins, axes = {}, {}
        for p0, q0, p1, q1, axis, column in self.joints:
            pos = pos + quat_apply(quat, p0.expand(count, -1))
            quat = quat_mul(quat, q0.expand(count, -1))
            if column is not None:
                local_axis = angles.new_zeros(count, 3)
                local_axis[:, axis] = 1
                origins[column] = pos
                axes[column] = quat_apply(quat, local_axis)
                rotation = angles.new_zeros(count, 4)
                rotation[:, 0] = (angles[:, column] / 2).cos()
                rotation[:, axis + 1] = (angles[:, column] / 2).sin()
                quat = quat_mul(quat, rotation)
            quat = quat_mul(quat, quat_conjugate(q1.expand(count, -1)))
            pos = pos - quat_apply(quat, p1.expand(count, -1))
        pos = pos + quat_apply(quat, self.offset_pos.expand(count, -1))
        quat = quat_mul(quat, self.offset_quat.expand(count, -1))
        if not jacobian:
            return pos, quat
        j = torch.stack([torch.cat((torch.cross(axes[i], pos - origins[i], dim=-1), axes[i]), -1)
                         for i in range(6)], -1)
        return pos, quat, j

    def solve(self, desired_pos, desired_quat, initial, limits, iterations=80,
              position_tolerance=0.0005, rotation_tolerance=0.003, *, _allow_cpu=True):
        # Small reset batches are launch/synchronization bound on CUDA. Keep the
        # same DLS iterations, seed, limits and tolerances on a cached CPU chain.
        if _allow_cpu and initial.is_cuda and len(initial) <= 128:
            if not hasattr(self, '_cpu_chain'):
                self._cpu_chain = object.__new__(ResetArmIK)
                self._cpu_chain.device = 'cpu'
                self._cpu_chain.joints = [
                    (*(value.cpu() for value in joint[:4]), *joint[4:]) for joint in self.joints]
                self._cpu_chain.offset_pos = self.offset_pos.cpu()
                self._cpu_chain.offset_quat = self.offset_quat.cpu()
            result = self._cpu_chain.solve(
                desired_pos.cpu(), desired_quat.cpu(), initial.cpu(), limits.cpu(),
                iterations, position_tolerance, rotation_tolerance)
            return tuple(value.to(initial.device) for value in result)
        angles = initial.clone()
        eye = torch.eye(6, device=angles.device, dtype=angles.dtype)
        for _ in range(iterations):
            pos, quat, jacobian = self.forward(angles, jacobian=True)
            rotation = axis_angle_from_quat(quat_mul(desired_quat, quat_conjugate(quat)))
            error = torch.cat((desired_pos - pos, rotation), -1)
            done = (error[:, :3].norm(dim=-1) < position_tolerance) & (rotation.norm(dim=-1) < rotation_tolerance)
            if bool(done.all()):
                break
            jt = jacobian.transpose(1, 2)
            delta = (jt @ torch.linalg.solve(jacobian @ jt + .02**2 * eye, error.unsqueeze(-1))).squeeze(-1)
            delta *= (.15 / delta.abs().amax(-1).clamp_min(.15)).unsqueeze(-1)
            angles = torch.maximum(torch.minimum(angles + torch.where(done[:, None], 0., delta), limits[..., 1]), limits[..., 0])
        pos, quat = self.forward(angles)
        pos_error = (pos - desired_pos).norm(dim=-1)
        rot_error = axis_angle_from_quat(quat_mul(desired_quat, quat_conjugate(quat))).norm(dim=-1)
        valid = torch.isfinite(angles).all(-1) & (pos_error < position_tolerance) & (rot_error < rotation_tolerance)
        return angles, valid, pos_error, rot_error
