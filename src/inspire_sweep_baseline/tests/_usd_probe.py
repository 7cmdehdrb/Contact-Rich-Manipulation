"""Run USD-only asset checks in a pxr-capable Python, without Isaac Sim."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom, UsdPhysics

module_root = Path(sys.argv[1]).resolve()
asset_root = module_root / "assets/data/ur5e_inspire_usd"
robot_stage = Usd.Stage.Open(str(asset_root / "ur5e_inspire.usd"))
assert robot_stage is not None
layers = [Path(layer.realPath).resolve() for layer in robot_stage.GetUsedLayers() if layer.realPath]
assert len(layers) == 5
assert all(path.is_relative_to(asset_root) and path.is_file() for path in layers)
joint_names = {
    prim.GetName() for prim in robot_stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
}
assert len(joint_names) == 18

stage = Usd.Stage.CreateInMemory()


def standard_cylinder_spawn(path, cfg, translation=None, orientation=None, **kwargs):
    """USD authoring shim for the simulator's standard cylinder spawner."""
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_props.mass)
    UsdGeom.Xform.Define(stage, f"{path}/geometry")
    mesh = UsdGeom.Cylinder.Define(stage, f"{path}/geometry/mesh")
    mesh.CreateHeightAttr(cfg.height)
    mesh.CreateRadiusAttr(cfg.radius)
    mesh.CreateAxisAttr("Z")
    return prim


source_path = module_root / "assets/mass_cylinder.py"
source_ast = ast.parse(source_path.read_text())
source_ast.body = [node for node in source_ast.body if isinstance(node, ast.FunctionDef)]
for node in source_ast.body:
    node.decorator_list = []
namespace = {
    "Gf": Gf, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics,
    "sim_utils": SimpleNamespace(spawn_cylinder=standard_cylinder_spawn),
}
exec(compile(source_ast, str(source_path), "exec"), namespace)
cfg = SimpleNamespace(radius=0.04, height=0.10, mass_props=SimpleNamespace(mass=1.0))
prim = namespace["spawn_mass_cylinder"]("/target", cfg)
mesh = stage.GetPrimAtPath("/target/geometry/mesh")
bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeRelativeBound(mesh, prim).ComputeAlignedRange()
mass = UsdPhysics.MassAPI(prim)
result = {
    "asset_layers": len(layers),
    "joint_names": sorted(joint_names),
    "bottom_z": bounds.GetMin()[2],
    "top_z": bounds.GetMax()[2],
    "mass": mass.GetMassAttr().Get(),
    "com_z": mass.GetCenterOfMassAttr().Get()[2],
    "inertia": list(mass.GetDiagonalInertiaAttr().Get()),
}
print(json.dumps(result))
