"""USD-only checks of right vendor geometry, inertias, joints and F/T assembly.

Run with a pxr-capable Python. No Isaac Sim/GPU is started.
"""
import ast
import __future__
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline'
path = ROOT / 'assets/data/wrench_right_hand/ur5e_inspire_right.usda'
stage = Usd.Stage.Open(str(path))
root = stage.GetDefaultPrim()
assert root.GetCustomDataByKey('handedness') == 'right'
metadata = json.loads((path.parent / 'tactile_regions.json').read_text())
assert root.GetCustomDataByKey('source_zip_sha256') == metadata['source_zip_sha256']
cache = UsdGeom.XformCache()
bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith('inspire_')]
assert len(bodies) == 30
assert abs(sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies) - .790) < 1e-7
for body in bodies:
    assert body.GetCustomDataByKey('vendor_right_link')
    colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1 and colliders[0].GetName() == 'right_geometry'
    mass = UsdPhysics.MassAPI(body)
    assert mass.GetMassAttr().Get() > 0
    assert all(v > 0 for v in mass.GetDiagonalInertiaAttr().Get())
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default']).ComputeUntransformedBound(body).ComputeAlignedRange()
    assert max(bound.GetSize()) < .25

joint_count = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint) or not prim.GetName().startswith('inspire_') or prim.GetName() == 'inspire_mount':
        continue
    joint_count += 1
    j = UsdPhysics.Joint(prim)
    matrices = []
    for side in (0, 1):
        body = stage.GetPrimAtPath(getattr(j, f'GetBody{side}Rel')().GetTargets()[0])
        pos, rot = getattr(j, f'GetLocalPos{side}Attr')().Get(), getattr(j, f'GetLocalRot{side}Attr')().Get()
        matrix = Gf.Matrix4d(1.)
        matrix.SetRotate(Gf.Quatd(rot)); matrix.SetTranslateOnly(Gf.Vec3d(pos))
        matrices.append(np.asarray(matrix * cache.GetLocalToWorldTransform(body)))
    np.testing.assert_allclose(matrices[0], matrices[1], atol=2e-6, rtol=0.)
assert joint_count == 29

# Execute the actual assembly's USD operations with only spawn/Sim leaves
# replaced. This verifies the right overlay preserves the measurement joint
# child frame expected by PureWrenchReader, without inventing sensor offsets.
assembled = Usd.Stage.CreateInMemory()
def spawn_usd(path, cfg, translation=None, orientation=None):
    prim = UsdGeom.Xform.Define(assembled, path).GetPrim()
    prim.GetReferences().AddReference(cfg.usd_path)
    return prim
def spawn_cylinder(path, cfg, translation=None, orientation=None):
    prim = UsdGeom.Xform.Define(assembled, path).GetPrim()
    tf = UsdGeom.Xformable(prim)
    tf.AddTranslateOp().Set(Gf.Vec3d(*translation))
    tf.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(cfg.mass_props.mass)
    return prim
def usd_cfg(**kw): return NS(**kw, func=spawn_usd)
def cylinder_cfg(**kw): return NS(**kw, func=spawn_cylinder)
sim = NS(UsdFileCfg=usd_cfg, CylinderCfg=cylinder_cfg, RigidBodyPropertiesCfg=NS,
         MassPropertiesCfg=NS, CollisionPropertiesCfg=NS, PreviewSurfaceCfg=NS)
tree = ast.parse((ROOT / 'assets/robot.py').read_text())
tree.body = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
for n in tree.body:
    if isinstance(n, ast.FunctionDef): n.decorator_list = []
ns = dict(Path=Path, __file__=str(ROOT/'assets/robot.py'), Gf=Gf, Sdf=Sdf, Usd=Usd,
          UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, sim_utils=sim, get_current_stage=lambda: assembled,
          ArticulationCfg=object)
exec(compile(tree, str(ROOT/'assets/robot.py'), 'exec'), ns)
cfg = NS(usd_path=str(path), rigid_props=None, articulation_props=None, sensor_height=.0254, sensor_radius=.041, sensor_mass=.001)
ns['spawn_ur5e_axia80_inspire']('/Robot', cfg)
joint = UsdPhysics.Joint(assembled.GetPrimAtPath('/Robot/joints/axia80_measurement_joint'))
assert str(joint.GetBody1Rel().GetTargets()[0]) == '/Robot/inspire_base_link'
np.testing.assert_allclose(joint.GetLocalPos1Attr().Get(), [0,0,0], atol=1e-6)
q = joint.GetLocalRot1Attr().Get()
assert abs(abs(q.GetReal()) - 1.) < 1e-6
assert assembled.GetPrimAtPath('/Robot/inspire_base_link').GetCustomDataByKey('vendor_right_link') == 'base_link'

# Real USD regression: a referenced rigid body without MassAPI must receive a
# mass; an existing child body must not acquire a second root rigid body.
tree = ast.parse((ROOT / 'assets/wrench_refactor_spawners.py').read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'set_shelf_mass']
ns = dict(Gf=Gf, Usd=Usd, UsdPhysics=UsdPhysics)
exec(compile(tree, 'wrench_refactor_spawners.py', 'exec'), ns)
schema_path = ROOT.parents[2] / 'IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py'
schema = ast.parse(schema_path.read_text())
modify = next(n for n in schema.body if isinstance(n, ast.FunctionDef) and n.name == 'modify_mass_properties')
modify.decorator_list = []
upstream = dict(UsdPhysics=UsdPhysics)
exec(compile(ast.Module(body=[modify], type_ignores=[]), str(schema_path), 'exec',
             flags=__future__.annotations.compiler_flag), upstream)
for child, original_mass in [(False, None), (True, None), (True, 100.)]:
    shelf_stage = Usd.Stage.CreateInMemory()
    shelf = UsdGeom.Xform.Define(shelf_stage, '/Shelf').GetPrim()
    body = UsdGeom.Xform.Define(shelf_stage, '/Shelf/body').GetPrim() if child else shelf
    UsdPhysics.RigidBodyAPI.Apply(body)
    if original_mass is None:
        # Actual Isaac Lab leaf returns before reading cfg: a mass override
        # alone cannot author the missing schema. This is the original defect.
        assert upstream['modify_mass_properties'](str(body.GetPath()), None, shelf_stage) is False
    if original_mass:
        api = UsdPhysics.MassAPI.Apply(body)
        api.CreateMassAttr(original_mass)
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(10., 20., 30.))
    ns['set_shelf_mass'](shelf, 30.)
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == 30.
    assert len([p for p in shelf_stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 1
    if original_mass:
        np.testing.assert_allclose(UsdPhysics.MassAPI(body).GetDiagonalInertiaAttr().Get(), [3.,6.,9.])
print(json.dumps({'right_bodies':len(bodies), 'aligned_right_joints':joint_count, 'paired_tactile_regions':len(metadata['regions']),
                  'hand_mass_kg':sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies), 'shelf_mass_regressions':3,
                  'measurement_child_identity_frame':True, 'scope':'USD-only, excludes PhysX contact and dynamic wrench cancellation'}))
