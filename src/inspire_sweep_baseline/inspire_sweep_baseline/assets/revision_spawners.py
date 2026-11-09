"""Revision assets: eight shared geometries and one physical body per slot.

Only the immutable geometry is shared. Each environment owns ten editable rigid
roots with independently sampled shape and uniform scale. Shape/scale stay fixed
after PhysX cooking; reset changes their poses and active obstacle membership.
"""

from collections.abc import Callable
import re
import random

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..mdp.common.contact_sweep_clutter_fusion import OBJECT_MASS
from .contact_sweep_clutter_fusion_holding_osc_object_random_spawners import spawn_prop
from .final_spawners import shelf_object_meshes
from .mass_cylinder import spawn_mass_cylinder
from .realistic_materials import bind_role_materials, material_defaults
from .robot import (
    HAND_CONTACT_BODY_NAMES, ROBOT_USD_PATH, SENSOR_HEIGHT, SENSOR_MASS,
    SENSOR_RADIUS, spawn_ur5e_axia80_inspire,
)


REVISION_OBJECT_KINDS = (
    "cylinder", "cube", "mug", "bottle", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "002_master_chef_can",
)
REVISION_OBJECT_NAMES = ("mass_cylinder", *(f"obstacle_{i}" for i in range(9)))
_prototype_cache = None


def _bind_shared_material(prim, role, friction, defaults):
    """Absolute bindings keep one material per role across all environment clones."""
    stage = prim.GetStage()
    bind_role_materials(stage, (str(prim.GetPath()),), f"/World/RevisionMaterials/{role}",
                        friction, material_defaults(defaults))


@clone
def spawn_revision_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_ur5e_axia80_inspire.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )
    bind_role_materials(prim.GetStage(), tuple(f"{prim_path}/{name}" for name in HAND_CONTACT_BODY_NAMES),
                        "/World/RevisionMaterials/Tool", .585, material_defaults(cfg.friction_defaults))
    return prim


@clone
def spawn_revision_shelf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim.spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation,
                                         orientation=orientation, **kwargs)
    _bind_shared_material(prim, "Floor", .315, cfg.friction_defaults)
    return prim


@configclass
class RevisionRobotSpawnerCfg(RigidObjectSpawnerCfg):
    """Independent asset configuration using the existing assembly function."""

    func: Callable = spawn_revision_robot
    usd_path: str = str(ROBOT_USD_PATH)
    activate_contact_sensors: bool = True
    rigid_props: sim.RigidBodyPropertiesCfg = sim.RigidBodyPropertiesCfg(
        disable_gravity=False, max_depenetration_velocity=5.0,
    )
    articulation_props: sim.ArticulationRootPropertiesCfg = sim.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
    )
    sensor_radius: float = SENSOR_RADIUS
    sensor_height: float = SENSOR_HEIGHT
    sensor_mass: float = SENSOR_MASS
    friction_defaults: sim.RigidBodyMaterialCfg = sim.RigidBodyMaterialCfg()


@configclass
class RevisionShelfSpawnerCfg(sim.UsdFileCfg):
    func: Callable = spawn_revision_shelf
    friction_defaults: sim.RigidBodyMaterialCfg = sim.RigidBodyMaterialCfg()


def _spawn_prototype(path, kind, cfg):
    """Build one normalized base shape, before the single random root scale."""
    stage = sim.get_current_stage()
    if kind == "cylinder":
        cylinder_cfg = sim.CylinderCfg(
            radius=.04, height=.10, axis="Z", mass_props=cfg.mass_props.copy(),
            rigid_props=cfg.rigid_props.copy(), collision_props=cfg.collision_props.copy(),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.1, .4, .8)),
            activate_contact_sensors=True,
        )
        root = spawn_mass_cylinder.__wrapped__(path, cylinder_cfg)
    elif kind in ("cube", "mug", "bottle"):
        root = sim.create_prim(path, "Xform")
        UsdGeom.Xform.Define(stage, path + "/Geometry")
        for index, (points, counts, indices) in enumerate(shelf_object_meshes(kind, (.06, .08, .10))):
            mesh_path = f"{path}/Geometry/Part_{index}"
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.3, .65, .75)])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            sim.modify_collision_properties(mesh_path, cfg.collision_props)
        sim.define_rigid_body_properties(path, cfg.rigid_props)
        sim.define_mass_properties(path, cfg.mass_props)
        sim.activate_contact_sensors(path)
    else:
        prop_cfg = cfg.copy()
        prop_cfg.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{kind}.usd"
        prop_cfg.size_fraction = .9
        prop_cfg.collision_kind = "box" if kind.endswith("_box") else "can"
        root = spawn_prop(path, prop_cfg)
    _bind_shared_material(root, "Object", .315, cfg.friction_defaults)
    # The source YCB helper creates a local physics material; all its bindings
    # have now been replaced by the shared material, so do not clone that prim.
    for child in tuple(root.GetChildren()):
        if child.GetName().startswith("RealisticObjectMaterial"):
            stage.RemovePrim(child.GetPath())
    return root


def _prototypes(stage, cfg):
    """Keep reusable USD layers off-stage: prototypes add no physical bodies."""
    global _prototype_cache
    stage_id = stage.GetRootLayer().identifier
    if _prototype_cache is not None and _prototype_cache["stage_id"] == stage_id:
        return _prototype_cache
    full_layer = Sdf.Layer.CreateAnonymous("revision_geometry.usda")
    root_layer = Sdf.Layer.CreateAnonymous("revision_roots.usda")
    bounds = []
    source_root = "/World/RevisionObjectTemplates"
    sim.create_prim(source_root, "Scope")
    try:
        for index, kind in enumerate(REVISION_OBJECT_KINDS):
            path = f"{source_root}/Shape_{index}"
            prim = _spawn_prototype(path, kind, cfg)
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeUntransformedBound(prim).ComputeAlignedRange()
            bound = np.asarray([tuple(box.GetMin()), tuple(box.GetMax())], dtype=np.float64)
            if not np.isfinite(bound).all() or not np.all(bound[1] > bound[0]):
                raise RuntimeError(f"Revision shape has invalid physical bounds: {kind}: {bound}")
            bounds.append(bound)
            # Absolute references to /World materials cannot cross the boundary
            # of an external geometry reference. Bind physics on each editable
            # destination root below instead of embedding out-of-scope targets.
            for descendant in Usd.PrimRange(prim):
                for relation in tuple(descendant.GetRelationships()):
                    if relation.GetName().startswith("material:binding:physics"):
                        descendant.RemoveProperty(relation.GetName())
                        remaining = descendant.GetRelationship(relation.GetName())
                        if remaining and any(str(target).startswith("/World/RevisionMaterials/")
                                             for target in remaining.GetTargets()):
                            raise RuntimeError(f"Shared geometry retains a world material binding: {descendant.GetPath()}")
            output_path = Sdf.Path(f"/Shape_{index}")
            Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(path), full_layer, output_path)
            Sdf.CopySpec(full_layer, output_path, root_layer, output_path)
    finally:
        stage.RemovePrim(source_root)
    # Roots are mutable. Geometry/collider descendants are external instances,
    # allowing USD and PhysX to share geometry while each root has its own scale.
    roots_stage = Usd.Stage.Open(root_layer)
    for index in range(len(REVISION_OBJECT_KINDS)):
        root = roots_stage.GetPrimAtPath(f"/Shape_{index}")
        for child in tuple(root.GetChildren()):
            child_path, type_name = child.GetPath(), child.GetTypeName()
            roots_stage.RemovePrim(child_path)
            shared = roots_stage.DefinePrim(child_path, type_name)
            shared.GetReferences().AddReference(full_layer.identifier, child_path)
            shared.SetInstanceable(True)
    _prototype_cache = {
        "stage_id": stage_id, "geometry": full_layer, "roots": root_layer,
        "bounds": np.asarray(bounds), "metadata": {},
    }
    return _prototype_cache


def spawn_revision_objects(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn one independent, uniformly scaled random shape at each matched path."""
    import carb

    del kwargs
    if tuple(cfg.scale_range) != (.9, 1.1):
        raise ValueError("Revision object scale range must be 0.9 to 1.1")
    stage = sim.get_current_stage()
    parent_expr, name = prim_path.rsplit("/", 1)
    parents = ([parent_expr] if re.fullmatch(r"[A-Za-z0-9/_]+", parent_expr)
               else sim.find_matching_prim_paths(parent_expr))
    if not parents:
        raise RuntimeError(f"No environment prims match {parent_expr}")
    prototypes = _prototypes(stage, cfg)
    paths = [f"{parent}/{name}" for parent in parents]
    choices = [random.randrange(len(REVISION_OBJECT_KINDS)) for _ in paths]
    # Consume the same draws before overriding so target-only experiments do
    # not shift the scale or obstacle random streams.
    if cfg.kind_override is not None:
        choices = [REVISION_OBJECT_KINDS.index(cfg.kind_override)] * len(paths)
    scales = [random.uniform(*cfg.scale_range) for _ in paths]
    group_count = getattr(cfg, 'reset_scale_groups', None)
    if group_count is not None:
        # Only Reset-Test opts in. Group by numeric env id, not USD traversal order.
        by_env = {int(re.search(r'/env_(\d+)/', path).group(1)): scale
                  for path, scale in zip(paths, scales)}
        scales = [by_env[int(re.search(r'/env_(\d+)/', path).group(1)) % group_count]
                  for path in paths]
    with Sdf.ChangeBlock():
        for path, kind in zip(paths, choices):
            Sdf.CreatePrimInLayer(stage.GetRootLayer(), path)
            Sdf.CopySpec(prototypes["roots"], Sdf.Path(f"/Shape_{kind}"), stage.GetRootLayer(), Sdf.Path(path))
    for path, kind, scale in zip(paths, choices, scales):
        root = stage.GetPrimAtPath(path)
        transform = UsdGeom.Xformable(root)
        transform.ClearXformOpOrder()
        # Source helpers may use a different precision for identity operations.
        # Remove those root-only operations before authoring a consistent pose.
        for attr in tuple(root.GetAttributes()):
            if attr.GetName().startswith("xformOp:"):
                root.RemoveProperty(attr.GetName())
        transform.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0., 0., 50.))))
        q = orientation or (1., 0., 0., 0.)
        transform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
        transform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale))
        mass = UsdPhysics.MassAPI(root)
        inertia = mass.GetDiagonalInertiaAttr().Get() if mass else None
        if inertia is not None and any(v > 0 for v in inertia):
            mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v) * scale ** 2 for v in inertia)))
        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            UsdShade.Material(stage.GetPrimAtPath("/World/RevisionMaterials/Object_0")),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics",
        )
        bound = prototypes["bounds"][kind] * scale
        root.SetCustomDataByKey("revision_kind", kind)
        root.SetCustomDataByKey("revision_scale", scale)
        prototypes["metadata"][path] = (kind, scale, bound)
    carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
    return stage.GetPrimAtPath(paths[0])


@configclass
class RevisionObjectSpawnerCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_revision_objects
    kind_override: str | None = None
    scale_range: tuple[float, float] = (.9, 1.1)
    activate_contact_sensors: bool = True
    mass_props: sim.MassPropertiesCfg = sim.MassPropertiesCfg(mass=OBJECT_MASS)
    rigid_props: sim.RigidBodyPropertiesCfg = sim.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16, solver_velocity_iteration_count=1,
        max_angular_velocity=1000., max_linear_velocity=1000., max_depenetration_velocity=5.,
        disable_gravity=True,
    )
    collision_props: sim.CollisionPropertiesCfg = sim.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=.001, rest_offset=0.,
    )
    friction_defaults: sim.RigidBodyMaterialCfg = sim.RigidBodyMaterialCfg()
    # Inputs used only by the reusable YCB normalization function.
    usd_path: str = ""
    size_fraction: float = .9
    collision_kind: str = "box"


def collect_revision_object_metadata(env, env_ids=None):
    """Collect already-scaled local bounds; do not apply a second random scale."""
    import torch

    if env.sim.is_playing():
        raise RuntimeError("Revision object geometry must be collected before PhysX starts")
    stage = sim.get_current_stage()
    if _prototype_cache is None or _prototype_cache["stage_id"] != stage.GetRootLayer().identifier:
        raise RuntimeError("Revision object prototypes have not been spawned on this stage")
    names = tuple(env.cfg.scene.object_collection.rigid_objects)
    if names != REVISION_OBJECT_NAMES:
        raise ValueError("Revision requires target slot 0 and exactly nine obstacle slots")
    ids = range(env.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
    if not hasattr(env, "_final_object_bounds"):
        env._final_object_bounds = np.zeros((env.num_envs, len(names), 2, 3), dtype=np.float64)
        env._final_object_scales = np.ones((env.num_envs, len(names)), dtype=np.float64)
        env._revision_object_kinds = np.zeros((env.num_envs, len(names)), dtype=np.int64)
    for row in ids:
        for index, name in enumerate(names):
            kind, scale, bounds = _prototype_cache["metadata"][f"/World/envs/env_{row}/{name}"]
            env._final_object_bounds[row, index] = bounds
            env._final_object_scales[row, index] = scale
            env._revision_object_kinds[row, index] = kind
    env._revision_base_object_bounds = _prototype_cache["bounds"].copy()
    env._revision_object_kind_names = REVISION_OBJECT_KINDS
