"""NVIDIA shelf props, uniformly scaled and bottom-origin rigid bodies."""

from .contact_sweep_clutter_fusion_holding_osc_object_random_geometry import fit_scale
from .contact_sweep_clutter_fusion_holding_osc_object_random_geometry import flat_collision_mesh
from .realistic_materials import bind_role_materials
from .realistic_materials import material_defaults
from collections.abc import Callable
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Gf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics
import isaaclab.sim as sim

ASSETS = ('003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '002_master_chef_can')


def spawn_prop(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage=sim.get_current_stage()
    root=sim.create_prim(prim_path, 'Xform', translation=translation, orientation=orientation)
    # Keep the reporting rigid body at the original bottom-origin slot path.
    wrapper=UsdGeom.Xform.Define(stage,prim_path+'/Geometry')
    upright=UsdGeom.Xform.Define(stage,prim_path+'/Geometry/Upright')
    # NVIDIA Axis_Aligned YCB props have their semantic top along -Y (verified against the package labels).
    upright.AddRotateXOp().Set(-90.0)
    visual=stage.DefinePrim(prim_path+'/Geometry/Upright/Asset','Xform')
    if not visual.GetReferences().AddReference(cfg.usd_path):
        raise RuntimeError(f'Cannot reference NVIDIA asset: {cfg.usd_path}')
    box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeRelativeBound(visual,root).ComputeAlignedRange()
    scale,offset=fit_scale(tuple(box.GetMin()),tuple(box.GetMax()),cfg.size_fraction)
    wrapper.AddTranslateOp().Set(Gf.Vec3d(*offset))
    wrapper.AddScaleOp().Set(Gf.Vec3f(scale,scale,scale))
    meshes=[]
    for prim in Usd.PrimRange(visual):
        if prim.IsInstance(): prim.SetInstanceable(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    if not meshes: raise RuntimeError(f'No meshes in {cfg.usd_path}')
    # Keep visual scans unchanged; construct one level-bottom collider in the
    # rigid root frame so scan rotations and uneven bottom vertices cannot rock it.
    dimensions = tuple(scale * (b - a) for a, b in zip(box.GetMin(), box.GetMax()))
    points, counts, indices = flat_collision_mesh(dimensions, cfg.collision_kind)
    collider_path = prim_path + '/Collision'
    collider = UsdGeom.Mesh.Define(stage, collider_path)
    collider.CreatePointsAttr(points)
    collider.CreateFaceVertexCountsAttr(counts)
    collider.CreateFaceVertexIndicesAttr(indices)
    collider.CreateSubdivisionSchemeAttr('none')
    collider.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr('convexHull')
    sim.define_rigid_body_properties(prim_path,cfg.rigid_props)
    sim.define_mass_properties(prim_path,cfg.mass_props)
    # Applying this to the root would re-enable colliders from the referenced scan.
    sim.modify_collision_properties(collider_path,cfg.collision_props)
    if cfg.activate_contact_sensors: sim.activate_contact_sensors(prim_path)
    bind_role_materials(stage,(prim_path,),prim_path+'/RealisticObjectMaterial',.315,material_defaults(sim.RigidBodyMaterialCfg()))
    root.SetCustomDataByKey('object_random_asset',cfg.usd_path)
    root.SetCustomDataByKey('object_random_uniform_scale',scale)
    root.SetCustomDataByKey('object_random_collision_kind',cfg.collision_kind)
    return root


@configclass
class ShelfPropCfg(sim.UsdFileCfg):
    func: Callable = spawn_prop
    size_fraction: float = 1.0
    collision_kind: str = 'box'


def prop_variants(original):
    return [ShelfPropCfg(
        usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{name}.usd',
        collision_kind='box' if name.endswith('_box') else 'can',
        size_fraction=fraction, mass_props=original.mass_props,
        rigid_props=original.rigid_props, collision_props=original.collision_props,
        activate_contact_sensors=True,
    ) for name in ASSETS for fraction in (.8,.9,1.)]
