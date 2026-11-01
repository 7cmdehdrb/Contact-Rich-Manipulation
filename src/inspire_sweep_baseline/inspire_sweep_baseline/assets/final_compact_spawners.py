"""v0_2_2-only prototype cache and globally shared physics materials.

Author bindings before cloning/PhysX startup. Runtime friction still uses the
parent's finite buckets and per-shape Tensor API, never edits USD bindings.
"""
from collections.abc import Callable
import hashlib
import random

import carb
from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade
import isaaclab.sim as sim
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from .realistic_spawners import spawn_realistic_robot, spawn_realistic_shelf

TEMPLATES = '/World/FinalCompactTemplates'
MATERIALS = '/World/FinalCompactMaterials'


def share_materials(prim):
    """Hoist complete physical-property signatures out of the cloned subtree."""
    stage = prim.GetStage()
    old_materials = []
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.MaterialAPI):
            old_materials.append(child.GetPath())
    for collider in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        source, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if not source:
            continue
        attrs = [(a.GetName(), a.GetTypeName(), a.Get()) for a in source.GetPrim().GetAttributes()
                 if a.GetName().startswith(('physics:', 'physxMaterial:')) and a.Get() is not None]
        key = hashlib.sha256(repr(sorted((name, str(kind), repr(value))
                                         for name, kind, value in attrs)).encode()).hexdigest()[:24]
        path = MATERIALS + '/m_' + key
        shared = UsdShade.Material.Get(stage, path)
        if not shared:
            shared = UsdShade.Material.Define(stage, path)
            UsdPhysics.MaterialAPI.Apply(shared.GetPrim())
            shared.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
            for name, kind, value in attrs:
                shared.GetPrim().CreateAttribute(name, kind, custom=False).Set(value)
        binding = collider
        while binding.IsInstanceProxy():
            binding = binding.GetParent()
        UsdShade.MaterialBindingAPI.Apply(binding).Bind(shared,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
    # Remove physical schemas from local copies while preserving any visual
    # shaders bound to the same USD material. Global physics materials survive.
    for path in old_materials:
        material = stage.GetPrimAtPath(path)
        material.RemoveAPI(UsdPhysics.MaterialAPI)
        material.RemoveAppliedSchema('PhysxMaterialAPI')


@clone
def spawn_shared_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_realistic_robot.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    share_materials(prim)
    return prim


@clone
def spawn_shared_shelf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_realistic_shelf.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    share_materials(prim)
    return prim


def spawn_cached_objects(prim_path, cfg, translation=None, orientation=None, **kwargs):
    stage = sim.get_current_stage()
    root, name = prim_path.rsplit('/', 1)
    roots = sim.find_matching_prim_paths(root)
    if not roots:
        raise ValueError(f'No environments for {prim_path}')
    prototypes = []
    for variant_id, source in zip(cfg.variant_ids, cfg.assets_cfg):
        path = TEMPLATES + f'/variant_{variant_id}'
        if not stage.GetPrimAtPath(path):
            source = source.copy()
            source.activate_contact_sensors = True
            source.func(path, source, translation=(0., 0., 0.), orientation=(1., 0., 0., 0.))
            share_materials(stage.GetPrimAtPath(path))
            stage.GetPrimAtPath(path).SetCustomDataByKey('compact_variant_id', variant_id)
        prototypes.append(path)
    layer = stage.GetRootLayer()
    with Sdf.ChangeBlock():
        for root_path in roots:
            destination = root_path + '/' + name
            Sdf.CreatePrimInLayer(layer, destination)
            Sdf.CopySpec(layer, Sdf.Path(random.choice(prototypes)), layer, Sdf.Path(destination))
            if translation is not None:
                attr = layer.GetAttributeAtPath(destination + '.xformOp:translate')
                if attr is not None:
                    attr.default = Gf.Vec3d(*translation)
    # Prototypes are discarded before simulation starts. Avoid per-env geometry
    # authoring/cooking; copied objects retain the configured parking pose.
    carb.settings.get_settings().set_bool('/isaaclab/spawn/multi_assets', True)
    return stage.GetPrimAtPath(roots[0] + '/' + name)


@configclass
class CachedFinalObjectsCfg(sim.MultiAssetSpawnerCfg):
    func: Callable = spawn_cached_objects
    variant_ids: tuple = ()
