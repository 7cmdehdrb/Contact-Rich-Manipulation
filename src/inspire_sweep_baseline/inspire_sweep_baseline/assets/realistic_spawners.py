"""Bind Average contact materials on the source asset before scene cloning.

Physics replication remains enabled. No MaterialBindingAPI is authored once
PhysX starts, and episode friction updates use only the Tensor API.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .mass_cylinder import MassCylinderCfg, spawn_mass_cylinder
from .realistic_materials import bind_role_materials, material_defaults
from .robot import HAND_CONTACT_BODY_NAMES, Ur5eAxia80InspireSpawnerCfg, spawn_ur5e_axia80_inspire


@clone
def spawn_realistic_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_ur5e_axia80_inspire.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )
    bind_role_materials(
        get_current_stage(), tuple(f"{prim_path}/{name}" for name in HAND_CONTACT_BODY_NAMES),
        f"{prim_path}/RealisticToolMaterial", 0.585, material_defaults(cfg.friction_defaults),
    )
    return prim


@clone
def spawn_realistic_shelf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_usd.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )
    bind_role_materials(
        get_current_stage(), (prim_path,), f"{prim_path}/RealisticFloorMaterial",
        0.315, material_defaults(cfg.friction_defaults),
    )
    return prim


@clone
def spawn_realistic_cylinder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_mass_cylinder.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )
    bind_role_materials(
        get_current_stage(), (prim_path,), f"{prim_path}/RealisticObjectMaterial",
        0.315, material_defaults(cfg.friction_defaults),
    )
    return prim


@configclass
class RealisticRobotSpawnerCfg(Ur5eAxia80InspireSpawnerCfg):
    func: Callable = spawn_realistic_robot
    friction_defaults: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()


@configclass
class RealisticShelfSpawnerCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_realistic_shelf
    friction_defaults: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()


@configclass
class RealisticCylinderSpawnerCfg(MassCylinderCfg):
    func: Callable = spawn_realistic_cylinder
    friction_defaults: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()


def replace_friction_spawner(original, spawner_type, defaults):
    """Preserve every source field apart from the spawn callback."""
    kwargs = {field.name: deepcopy(getattr(original, field.name))
              for field in fields(original) if field.name not in ("func", "friction_defaults")}
    return spawner_type(**kwargs, friction_defaults=defaults.copy())
