#!/usr/bin/env python3
"""Build or check observation-only tactile regions from the unchanged robot USD.

This CPU utility needs pxr, NumPy, and SciPy, but does not import Isaac Sim or
Isaac Lab. It only reads source geometry. The output describes virtual sensing
regions and must never be used to author collision meshes or robot properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "inspire_sweep_baseline" / "assets" / "data"
DEFAULT_OUTPUT = ASSET_DIRECTORY / "tactile_regions.json"
NAMES = (
    "inspire_palm_force_sensor",
    *(f"inspire_thumb_force_sensor_{index}" for index in range(1, 5)),
    *(
        f"inspire_{finger}_force_sensor_{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(1, 4)
    ),
)
VIRTUAL_MARGIN_M = 0.0005


def build_metadata() -> dict:
    """Read the composed USD and return reproducible virtual region metadata."""
    source = ASSET_DIRECTORY / "ur5e_inspire_usd" / "ur5e_inspire.usd"
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open robot USD: {source}")
    cache = UsdGeom.XformCache()

    def collision_points_in_frame(body, frame):
        colliders = [
            prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(colliders) != 1:
            raise RuntimeError(f"Expected one collider for {body.GetPath()}")
        collider = colliders[0]
        if not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get():
            raise RuntimeError(f"Disabled collider: {collider.GetPath()}")
        if UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() != "convexHull":
            raise RuntimeError(f"Expected convexHull collider: {collider.GetPath()}")
        meshes = [
            prim for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies())
            if prim.IsA(UsdGeom.Mesh)
        ]
        if len(meshes) != 1:
            raise RuntimeError(f"Expected one collision mesh: {collider.GetPath()}")
        mesh = meshes[0]
        vertices = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(
            cache.GetLocalToWorldTransform(mesh) * cache.GetLocalToWorldTransform(frame).GetInverse()
        )
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
        return np.unique(vertices, axis=0)

    joints = {}
    actual_sensor_names = set()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "force_sensor" in prim.GetName():
            actual_sensor_names.add(prim.GetName())
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(prim)
            children = joint.GetBody1Rel().GetTargets()
            if len(children) == 1 and children[0].name in NAMES:
                parents = joint.GetBody0Rel().GetTargets()
                if len(parents) != 1 or children[0].name in joints:
                    raise RuntimeError(f"Ambiguous sensor parent: {prim.GetPath()}")
                joints[children[0].name] = parents[0]
    if actual_sensor_names != set(NAMES) or set(joints) != set(NAMES):
        raise RuntimeError("Robot USD must contain the expected 17 fixed-jointed tactile bodies")

    regions = []
    for name in NAMES:
        sensor = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild(name))
        parent = stage.GetPrimAtPath(joints[name])
        if not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Sensor parent is not a rigid body: {parent.GetPath()}")
        pad = collision_points_in_frame(sensor, sensor)
        parent_vertices = collision_points_in_frame(parent, sensor)
        footprint = ConvexHull(pad[:, :2])
        pad_hull = ConvexHull(pad)
        parent_hull = ConvexHull(parent_vertices)
        top_facets = pad_hull.equations[:, 2] >= 0.5
        if not top_facets.any():
            raise RuntimeError(f"No outward-facing pad facets: {name}")
        top_vertices = np.unique(pad_hull.simplices[top_facets])
        z_min = float(pad[top_vertices, 2].min() - VIRTUAL_MARGIN_M)
        z_max = float(parent_vertices[:, 2].max() + VIRTUAL_MARGIN_M)
        if z_max <= z_min:
            raise RuntimeError(f"Invalid virtual region depth bounds: {name}")

        # Intersect a +Z ray with the authored parent convex hull. This proves
        # each virtual footprint includes a real parent collision surface.
        xy_vertices = pad[footprint.vertices, :2]
        candidates = [np.zeros(2), xy_vertices.mean(axis=0), *xy_vertices]
        planes = parent_hull.equations
        support = None
        for xy in candidates:
            if (footprint.equations[:, :2] @ xy + footprint.equations[:, 2]).max() > 1e-8:
                continue
            c = planes[:, :2] @ xy + planes[:, 3]
            nz = planes[:, 2]
            if np.any(c[np.abs(nz) <= 1e-10] > 1e-8):
                continue
            upper = np.min(-c[nz > 1e-10] / nz[nz > 1e-10])
            lower = np.max(-c[nz < -1e-10] / nz[nz < -1e-10])
            if lower > upper + 1e-8 or not (z_min <= upper <= z_max):
                continue
            support = [float(xy[0]), float(xy[1]), float(upper)]
            if np.max(planes[:, :3] @ np.asarray(support) + planes[:, 3]) >= 1e-7:
                raise RuntimeError(f"Parent surface probe lies outside convex hull: {name}")
            break
        if support is None:
            raise RuntimeError(f"No parent collision surface inside virtual region: {name}")
        support_errors = planes[:, :3] @ np.asarray(support) + planes[:, 3]
        support_normal = planes[support_errors.argmax(), :3]
        maximum_separation = float((pad @ planes[:, :3].T + planes[:, 3]).max())
        regions.append({
            "name": name,
            "parent_name": parent.GetName(),
            "footprint_halfspaces": np.unique(np.round(footprint.equations, 12), axis=0).tolist(),
            "pad_bounds_m": {"min": pad.min(axis=0).tolist(), "max": pad.max(axis=0).tolist()},
            "outward_z_min_m": z_min,
            "outward_z_max_m": z_max,
            "normal_axis": [0.0, 0.0, 1.0],
            "verification": {
                "parent_surface_point_local_m": support,
                "parent_surface_outward_z_m": support[2],
                "parent_surface_outward_normal_local": support_normal.tolist(),
                "parent_surface_gap_to_pad_max_z_m": support[2] - float(pad[:, 2].max()),
                "pad_max_signed_parent_plane_distance_m": maximum_separation,
                "pad_wholly_buried_in_parent_convex_hull": bool(maximum_separation < -1e-6),
                "footprint_edge_count": len(footprint.equations),
            },
        })

    layers = sorted({Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath})
    return {
        "schema_version": 1,
        "coordinate_frame": "sensor_body_local",
        "outward_axis": [0.0, 0.0, 1.0],
        "virtual_margin_m": VIRTUAL_MARGIN_M,
        "source": {
            "usd_path": str(source.relative_to(ASSET_DIRECTORY)),
            "layer_sha256": {
                str(path.relative_to(ASSET_DIRECTORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in layers
            },
        },
        "region_definition": (
            "Footprint: 2D convex hull of the existing active collision mesh projected onto sensor-body local XY. "
            "Outward direction: sensor-body local +Z (thin sensor normal). "
            "Minimum Z: lowest vertex on an outward-facing pad hull facet with normal dot +Z >= 0.5, "
            "minus 0.5 mm observation-only margin. Maximum Z: greatest parent collision vertex Z in the same "
            "sensor-body frame plus 0.5 mm observation-only margin. Thus existing broad parent convex hull "
            "contacts over a buried pad can activate its channel without moving or creating collision geometry. "
            "The runtime should additionally reject side/back contacts using the outward normal direction. "
            "These are virtual localized contact regions, not a pressure/deformation or load-transfer model."
        ),
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Metadata JSON destination")
    parser.add_argument("--check", action="store_true", help="Compare regenerated metadata without writing")
    args = parser.parse_args()
    metadata = build_metadata()
    if args.check:
        if not args.output.is_file():
            parser.exit(1, f"Missing metadata: {args.output}\n")
        if json.loads(args.output.read_text()) != metadata:
            parser.exit(1, f"Tactile metadata differs from regenerated USD regions: {args.output}\n")
        parent_count = len({region["parent_name"] for region in metadata["regions"]})
        print(
            f"Tactile region check passed: {len(metadata['regions'])} regions, "
            f"{parent_count} parents, source hashes match"
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {args.output}: {len(metadata['regions'])} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
