"""Startup-only USD collider bounds for final reset certification."""

from __future__ import annotations

from itertools import product

from pxr import Gf, Usd, UsdGeom, UsdPhysics
import torch

from .action_math import quaternion_rotate, quaternion_to_matrix
from .contact_math import oriented_box_overlap


def _collision_shapes(body):
    paths = set()
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if UsdGeom.Boundable(child):
                    paths.add(str(child.GetPath()))
    return sorted(paths)


def _points_in_frame(stage, body, reference, *, mesh_vertices=False):
    cache = UsdGeom.XformCache()
    reference_inverse = cache.GetLocalToWorldTransform(reference).GetInverse()
    points = []
    for path in _collision_shapes(body):
        shape = stage.GetPrimAtPath(path)
        transform = cache.GetLocalToWorldTransform(shape) * reference_inverse
        if mesh_vertices and shape.IsA(UsdGeom.Mesh):
            vertices = UsdGeom.Mesh(shape).GetPointsAttr().Get()
            points.extend(transform.Transform(Gf.Vec3d(*vertex)) for vertex in vertices)
            continue
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(shape), Usd.TimeCode.Default())
        if extent is None or len(extent) != 2:
            raise RuntimeError(f"Collider {path} has no computed geometric extent")
        # Plugin-computed bounds avoid procedural USD fallback [-1,1] extents.
        points.extend(transform.Transform(Gf.Vec3d(*corner)) for corner in product(
            (float(extent[0][0]), float(extent[1][0])),
            (float(extent[0][1]), float(extent[1][1])),
            (float(extent[0][2]), float(extent[1][2])),
        ))
    if not points:
        raise RuntimeError(f"Rigid body {body.GetPath()} has no collision geometry")
    return points


class RobotCollisionBounds:
    """Cache all robot collision meshes as conservative body-local boxes."""

    def __init__(self, stage, robot_root_path, body_names, device):
        centers, half_extents = [], []
        for name in body_names:
            body = stage.GetPrimAtPath(robot_root_path + "/" + name)
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing collision-bound rigid body {body.GetPath()}")
            points = _points_in_frame(stage, body, body)
            minimum = [min(float(p[axis]) for p in points) for axis in range(3)]
            maximum = [max(float(p[axis]) for p in points) for axis in range(3)]
            centers.append([(lo + hi) * .5 for lo, hi in zip(minimum, maximum)])
            half_extents.append([(hi - lo) * .5 for lo, hi in zip(minimum, maximum)])
        self.centers = torch.tensor(centers, device=device, dtype=torch.float32)
        self.half_extents = torch.tensor(half_extents, device=device, dtype=torch.float32)
        if not torch.isfinite(self.centers).all() or not torch.isfinite(self.half_extents).all() or (self.half_extents <= 0).any():
            raise RuntimeError("Robot collision bounds must be finite and nondegenerate")
        hand = stage.GetPrimAtPath(robot_root_path + "/inspire_base_link")
        pad = stage.GetPrimAtPath(robot_root_path + "/inspire_palm_force_sensor")
        points = _points_in_frame(stage, pad, hand, mesh_vertices=True)
        minimum = [min(float(p[axis]) for p in points) for axis in range(3)]
        maximum = [max(float(p[axis]) for p in points) for axis in range(3)]
        # Actual exposed mesh front, in H axes; do not aim from the wrist origin.
        self.palm_face_h = torch.tensor(
            [(minimum[0] + maximum[0]) * .5, maximum[1], (minimum[2] + maximum[2]) * .5],
            device=device, dtype=torch.float32,
        )

    def overlaps(self, body_pos, body_quat, obstacle_pos, obstacle_quat, obstacle_size, margin):
        center_w = body_pos + quaternion_rotate(body_quat, self.centers.expand_as(body_pos))
        return oriented_box_overlap(
            center_w, quaternion_to_matrix(body_quat), self.half_extents.unsqueeze(0) + margin,
            obstacle_pos.unsqueeze(1), quaternion_to_matrix(obstacle_quat).unsqueeze(1),
            body_pos.new_tensor(obstacle_size).reshape(1, 1, 3) * .5,
        )
