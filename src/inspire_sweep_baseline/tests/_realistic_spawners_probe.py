"""Run production spawn-time binding functions against USD-only delegates."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("materials", root / "assets/realistic_materials.py")
materials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materials)
stage = Usd.Stage.CreateInMemory()
defaults = SimpleNamespace(restitution=0.0, restitution_combine_mode="average",
                           compliant_contact_stiffness=0.0, compliant_contact_damping=0.0)
names = []
calls = []


def base_robot(path, cfg, **kwargs):
    calls.append(("robot", kwargs))
    robot = UsdGeom.Xform.Define(stage, path).GetPrim()
    robot.GetReferences().AddReference(str(root / "assets/data/ur5e_inspire_usd/ur5e_inspire.usd"))
    names.extend(prim.GetName() for prim in robot.GetChildren()
                 if prim.GetName().startswith("inspire_") and prim.HasAPI(UsdPhysics.RigidBodyAPI))
    return robot


def base_shelf(path, cfg, **kwargs):
    calls.append(("shelf", kwargs))
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"{path}/collision").GetPrim())
    return prim


def standard_cylinder_spawn(path, cfg, **kwargs):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_props.mass)
    UsdGeom.Xform.Define(stage, f"{path}/geometry")
    mesh = UsdGeom.Cylinder.Define(stage, f"{path}/geometry/mesh")
    mesh.CreateHeightAttr(cfg.height)
    mesh.CreateRadiusAttr(cfg.radius)
    mesh.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return prim


# Execute the actual bottom-origin/COM/inertia authoring code too.
mass_ast = ast.parse((root / "assets/mass_cylinder.py").read_text())
mass_ast.body = [node for node in mass_ast.body if isinstance(node, ast.FunctionDef)]
for node in mass_ast.body:
    node.decorator_list = []
mass_ns = {"Gf": Gf, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics,
           "sim_utils": SimpleNamespace(spawn_cylinder=standard_cylinder_spawn)}
exec(compile(mass_ast, "mass_cylinder.py", "exec"), mass_ns)


def base_cylinder(path, cfg, **kwargs):
    calls.append(("object", kwargs))
    return mass_ns["spawn_mass_cylinder"](path, cfg, **kwargs)


spawn_ast = ast.parse((root / "assets/realistic_spawners.py").read_text())
spawn_ast.body = [node for node in spawn_ast.body if isinstance(node, ast.FunctionDef) and node.name.startswith("spawn_")]
for node in spawn_ast.body:
    node.decorator_list = []
namespace = {
    "get_current_stage": lambda: stage,
    "bind_role_materials": materials.bind_role_materials,
    "material_defaults": materials.material_defaults,
    "HAND_CONTACT_BODY_NAMES": names,
    "spawn_ur5e_axia80_inspire": SimpleNamespace(__wrapped__=base_robot),
    "spawn_mass_cylinder": SimpleNamespace(__wrapped__=base_cylinder),
    "sim_utils": SimpleNamespace(spawn_from_usd=SimpleNamespace(__wrapped__=base_shelf)),
}
exec(compile(spawn_ast, "realistic_spawners.py", "exec"), namespace)
cfg = SimpleNamespace(friction_defaults=defaults, radius=0.04, height=0.10, mass_props=SimpleNamespace(mass=0.365))
robot = namespace["spawn_realistic_robot"]("/World/envs/env_0/Robot", cfg, translation=(1., 2., 3.), orientation=(1., 0., 0., 0.))
shelf = namespace["spawn_realistic_shelf"]("/World/envs/env_0/Shelf", cfg)
obj = namespace["spawn_realistic_cylinder"]("/World/envs/env_0/mass_cylinder", cfg)
assert calls[0][1]["translation"] == (1., 2., 3.)
assert calls[0][1]["orientation"] == (1., 0., 0., 0.)
for prim, mean in ((robot, 0.585), (shelf, 0.315), (obj, 0.315)):
    colliders = [child for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                 if child.HasAPI(UsdPhysics.CollisionAPI) and (prim != robot or "inspire_" in str(child.GetPath()))]
    assert colliders
    for collider in colliders:
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert material and str(material.GetPath()).startswith(str(prim.GetPath()))
        assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "average"
        assert abs(UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get() - mean) < 1e-6
mass = UsdPhysics.MassAPI(obj)
assert abs(mass.GetMassAttr().Get() - 0.365) < 1e-6
assert abs(mass.GetCenterOfMassAttr().Get()[2] - 0.02) < 1e-6
mesh = stage.GetPrimAtPath(f"{obj.GetPath()}/geometry/mesh")
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeRelativeBound(mesh, obj).ComputeAlignedRange()
assert abs(bbox.GetMin()[2]) < 1e-6 and abs(bbox.GetMax()[2] - 0.1) < 1e-6
print(json.dumps({"delegated_roles": [role for role, _ in calls], "mean_mass": 0.365, "com_z": 0.02, "hand_bodies": len(names)}))
