"""Verify real copied Hand instance colliders and reusable per-env materials."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

module_root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("realistic_materials", module_root / "assets/realistic_materials.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stage = Usd.Stage.CreateInMemory()
defaults = module.material_defaults(SimpleNamespace(
    restitution=0.0, restitution_combine_mode="average",
    compliant_contact_stiffness=0.0, compliant_contact_damping=0.0,
))
robots, bindings = [], []
for index in range(2):
    env_prim = UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}").GetPrim()
    if index == 0:
        robot = UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}/Robot").GetPrim()
        robot.GetReferences().AddReference(str(module_root / "assets/data/ur5e_inspire_usd/ur5e_inspire.usd"))
    else:
        # Match InteractiveScene's clone_environments(copy_from_source=False).
        env_prim.GetInherits().AddInherit("/World/envs/env_0")
        robot = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
    body_paths = [str(prim.GetPath()) for prim in robot.GetChildren()
                  if prim.GetName().startswith("inspire_") and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    bindings.append(module.bind_role_materials(stage, body_paths, f"/World/Private/env_{index}/tool", 0.585, defaults))
    robots.append(robot)
assert len(bindings[0]) == len(bindings[1]) == 1
for robot in robots:
    colliders = [prim for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI) and "inspire_" in str(prim.GetPath())]
    assert len(colliders) == 30
    for collider in colliders:
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert material
        assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "average"
count_before = sum(prim.IsA(UsdShade.Material) for prim in stage.Traverse())
for sample in range(10):
    bindings[0][0][0].Set(0.2 + sample * 0.01)
    bindings[0][0][1].Set((0.2 + sample * 0.01) * 0.8)
assert bindings[1][0][0].Get() > 0.58
assert count_before == sum(prim.IsA(UsdShade.Material) for prim in stage.Traverse())

# Changing friction must retain an existing material's non-friction properties.
mesh = UsdGeom.Cube.Define(stage, "/World/probe/collision").GetPrim()
UsdPhysics.CollisionAPI.Apply(mesh)
old = UsdShade.Material.Define(stage, "/World/original")
api = UsdPhysics.MaterialAPI.Apply(old.GetPrim())
api.CreateRestitutionAttr(0.23)
old.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
old.GetPrim().CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(13.0)
UsdShade.MaterialBindingAPI.Apply(mesh).Bind(old, materialPurpose="physics")
module.bind_role_materials(stage, ["/World/probe"], "/World/private/probe", 0.315, defaults)
new, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics")
assert abs(UsdPhysics.MaterialAPI(new.GetPrim()).GetRestitutionAttr().Get() - 0.23) < 1e-6
assert new.GetPrim().GetAttribute("physxMaterial:restitutionCombineMode").Get() == "max"
assert new.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness").Get() == 13.0
print(json.dumps({"hand_colliders_per_env": 30, "private_tool_materials": 2, "count_stable": True, "restitution_preserved": True}))
