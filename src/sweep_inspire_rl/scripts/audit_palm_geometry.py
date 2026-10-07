#!/usr/bin/env python3
"""Audit the source USD's convex collision hulls using pxr, NumPy and SciPy.

Run in a USD-enabled Python environment. No physics simulation is required.
The geometry is measured in inspire_base_link coordinates; its +Y axis is
world +Y at this task's fixed right-facing orientation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(path: Path) -> dict:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    cache = UsdGeom.XformCache()

    def body(name):
        matches = [p for p in stage.Traverse() if p.GetName() == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one body {name}, got {len(matches)}")
        return matches[0]

    hand_inverse = cache.GetLocalToWorldTransform(body("inspire_base_link")).GetInverse()

    def collider_points(name):
        meshes = {}
        for collider in Usd.PrimRange(body(name), Usd.TraverseInstanceProxies()):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(collider)
            if api.GetCollisionEnabledAttr().Get() is False:
                continue
            approximation = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
            if approximation != "convexHull":
                raise RuntimeError(f"Expected convexHull at {collider.GetPath()}, got {approximation}")
            for mesh in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
                if mesh.IsA(UsdGeom.Mesh):
                    meshes[str(mesh.GetPath())] = mesh
        if len(meshes) != 1:
            raise RuntimeError(f"Expected one collision mesh for {name}, got {len(meshes)}")
        mesh = next(iter(meshes.values()))
        transform = cache.GetLocalToWorldTransform(mesh) * hand_inverse
        return np.array([
            transform.Transform(Gf.Vec3d(*point))
            for point in UsdGeom.Mesh(mesh).GetPointsAttr().Get()
        ])

    base_points = collider_points("inspire_base_link")
    pad_points = collider_points("inspire_palm_force_sensor")
    base_hull = ConvexHull(base_points).equations
    pad_hull = ConvexHull(pad_points).equations

    def ray_interval(equations, x, z):
        a = equations[:, 1]
        b = equations[:, 0] * x + equations[:, 2] * z + equations[:, 3]
        lower = max(-b[a < -1e-9] / a[a < -1e-9], default=-float("inf"))
        upper = min(-b[a > 1e-9] / a[a > 1e-9], default=float("inf"))
        if lower > upper or not np.all(b[np.abs(a) <= 1e-9] <= 1e-8):
            raise RuntimeError(f"Ray at H x={x}, z={z} misses a collider")
        return [float(lower), float(upper)]

    rays = []
    for x, z in ((0.0004, 0.1196), (-0.02, 0.1196), (0.02, 0.1196)):
        base = ray_interval(base_hull, x, z)
        pad = ray_interval(pad_hull, x, z)
        rays.append({
            "x_h_m": x, "z_h_m": z, "base_y_interval_h_m": base,
            "pad_y_interval_h_m": pad, "base_ahead_of_pad_m": base[1] - pad[1],
        })
    signed_distances = pad_points @ base_hull[:, :3].T + base_hull[:, 3]
    relative_origins = {}
    for name in ("inspire_palm_force_sensor", "inspire_middle_force_sensor_3", "inspire_little_force_sensor_3"):
        transform = cache.GetLocalToWorldTransform(body(name)) * hand_inverse
        relative_origins[name] = list(transform.ExtractTranslation())
    return {
        "robot_usd": str(path), "robot_usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "collision_approximation": "convexHull", "origins_in_hand_frame_m": relative_origins,
        "pad_vertices_inside_base_hull_fraction": float(np.mean(np.max(signed_distances, axis=1) <= 1e-8)),
        "pad_max_signed_distance_to_base_hull_m": float(np.max(signed_distances)),
        "rays": rays,
        "limitation": "Static source collision geometry; does not reproduce the remote checkpoint or cooked PhysX contacts.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", type=Path, default=Path(__file__).resolve().parents[2] / "hand_manipulation_rl/hand_manipulation_rl/assets/data/ur5e_inspire_usd/ur5e_inspire.usd")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.robot_usd.resolve())
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
