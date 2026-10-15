"""Exercise production tactile localization in the original USD sensor frames.

Runs in a CPU Python with pxr and torch. Only the production mapper class is
loaded through AST to avoid simulator imports; its implementation and Isaac
Lab's quaternion functions execute unchanged. No USD or metadata is written.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    module_root = Path(sys.argv[1]).resolve()
    asset_directory = module_root / "assets/data"
    metadata = json.loads((asset_directory / "tactile_regions.json").read_text())
    regions = metadata["regions"]
    expected_hashes = metadata["source"]["layer_sha256"]

    def asset_hashes():
        return {
            name: hashlib.sha256((asset_directory / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }

    before = asset_hashes()
    assert before == expected_hashes, "Original USD differs from the calibrated sensor geometry"
    stage = Usd.Stage.Open(str(asset_directory / metadata["source"]["usd_path"]))
    assert stage is not None
    cache = UsdGeom.XformCache()
    names = [region["name"] for region in regions]
    actual_names = {
        prim.GetName() for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "force_sensor" in prim.GetName()
    }
    assert actual_names == set(names)

    # Preserve every neighboring sensor's actual authored body frame.
    positions, quaternions, transforms = [], [], []
    for name in names:
        body = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild(name))
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        matrix = cache.GetLocalToWorldTransform(body)
        transforms.append(matrix)
        transform = Gf.Transform(matrix)
        position = transform.GetTranslation()
        rotation = transform.GetRotation().GetQuat().GetNormalized()
        positions.append(list(position))
        quaternions.append([rotation.GetReal(), *rotation.GetImaginary()])
    positions = torch.tensor([positions], dtype=torch.float32)
    quaternions = torch.tensor([quaternions], dtype=torch.float32)

    math_path = module_root.parents[2] / "IsaacLab/source/isaaclab/isaaclab/utils/math.py"
    official_math = load_module("_tactile_geometry_official_math", math_path)
    mapper_path = module_root / "mdp/tactile_observations.py"
    source_ast = ast.parse(mapper_path.read_text())
    mapper_classes = [
        node for node in source_ast.body
        if isinstance(node, ast.ClassDef) and node.name == "ParentContactRegions"
    ]
    assert len(mapper_classes) == 1
    namespace = {"torch": torch, "quat_apply_inverse": official_math.quat_apply_inverse}
    mapper_ast = ast.Module(body=mapper_classes, type_ignores=[])
    exec(compile(mapper_ast, str(mapper_path), "exec"), namespace)
    mapper_class = namespace["ParentContactRegions"]
    sensor_math = load_module("_tactile_geometry_sensor_math", module_root / "mdp/sensors.py")

    parent_names = list(dict.fromkeys(region["parent_name"] for region in regions))
    mappers = {}
    for parent in parent_names:
        channels = [index for index, region in enumerate(regions) if region["parent_name"] == parent]
        mappers[parent] = mapper_class([regions[index] for index in channels], channels, "cpu")

    world_points, inward_normals = [], []
    for region, matrix in zip(regions, transforms):
        verification = region["verification"]
        point = matrix.Transform(Gf.Vec3d(*verification["parent_surface_point_local_m"]))
        outward = matrix.TransformDir(Gf.Vec3d(*verification["parent_surface_outward_normal_local"]))
        outward.Normalize()
        world_points.append(list(point))
        inward_normals.append(list(-outward))
    world_points = torch.tensor(world_points, dtype=torch.float32)
    inward_normals = torch.tensor(inward_normals, dtype=torch.float32)

    activated_channels = []
    for channel, region in enumerate(regions):
        output = torch.zeros(1, len(regions), 3)
        normal = inward_normals[channel:channel + 1]
        mappers[region["parent_name"]].accumulate(
            output, world_points[channel:channel + 1], normal * 0.1, normal,
            torch.zeros(1, dtype=torch.long), positions, quaternions,
        )
        _, bits = sensor_math.contact_magnitudes_and_bits(output)
        activated = bits[0].nonzero().flatten().tolist()
        assert activated == [channel], (
            f"Original USD frames mapped {region['name']} probe to {activated}, expected [{channel}]"
        )
        activated_channels.extend(activated)

    # Simultaneous contacts retain all neighboring channels on shared parents.
    combined = torch.zeros(1, len(regions), 3)
    for parent, mapper in mappers.items():
        channels = [index for index, region in enumerate(regions) if region["parent_name"] == parent]
        mapper.accumulate(
            combined, world_points[channels], inward_normals[channels] * 0.1, inward_normals[channels],
            torch.zeros(len(channels), dtype=torch.long), positions, quaternions,
        )
    _, combined_bits = sensor_math.contact_magnitudes_and_bits(combined)
    simultaneous = combined_bits[0].nonzero().flatten().tolist()
    assert simultaneous == list(range(len(regions)))
    after = asset_hashes()
    assert before == after, "Geometry probe modified the original USD layers"
    print(json.dumps({
        "region_count": len(regions),
        "parent_count": len(parent_names),
        "activated_channels": activated_channels,
        "simultaneous_activated_channels": simultaneous,
        "neighbor_frames": "original_usd",
        "source_hashes_unchanged": before == after,
        "metadata_source_hashes_match": before == expected_hashes,
    }))


if __name__ == "__main__":
    main()
