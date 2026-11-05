"""Refactor shelf: establish one rigid body before authoring its total mass."""

from collections.abc import Callable
from pxr import Gf, Usd, UsdPhysics
import isaaclab.sim as sim
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from .revision_spawners import _bind_shared_material


def set_shelf_mass(prim, mass_kg):
    """Author MassAPI on the actual body, including assets with no mass schema.

    A RigidObject must have one body. Do not assign 30 kg to every collider or
    silently accept a multi-body shelf. Preserve existing dynamic/kinematic flags.
    """
    bodies = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) > 1:
        raise ValueError(f"Shelf must have one rigid body, found {len(bodies)}")
    body = bodies[0] if bodies else prim
    if not bodies:
        UsdPhysics.RigidBodyAPI.Apply(body)
    api = UsdPhysics.MassAPI.Apply(body)
    old_mass = api.GetMassAttr().Get()
    inertia = api.GetDiagonalInertiaAttr().Get()
    if inertia is not None and any(float(v) > 0 for v in inertia):
        # Keep authored inertia consistent with the new mass, or ask PhysX to
        # compute it from colliders when the original mass was density-derived.
        scale = mass_kg / old_mass if old_mass and old_mass > 0 else 0.
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) * scale for v in inertia)))
    api.CreateMassAttr(float(mass_kg))
    return body


@clone
def spawn_refactor_shelf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    # spawn_from_usd only MODIFIES existing MassAPI schemas. Defer the override
    # until after the reference and rigid-body hierarchy actually exist.
    reference_cfg = cfg.copy()
    reference_cfg.mass_props = None
    prim = sim.spawn_from_usd.__wrapped__(
        prim_path, reference_cfg, translation=translation, orientation=orientation, **kwargs)
    set_shelf_mass(prim, cfg.mass_props.mass)
    _bind_shared_material(prim, "Floor", .315, cfg.friction_defaults)
    return prim


@configclass
class WrenchRefactorShelfSpawnerCfg(sim.UsdFileCfg):
    func: Callable = spawn_refactor_shelf
    mass_props: sim.MassPropertiesCfg = sim.MassPropertiesCfg(mass=30.)
    friction_defaults: sim.RigidBodyMaterialCfg = sim.RigidBodyMaterialCfg()
