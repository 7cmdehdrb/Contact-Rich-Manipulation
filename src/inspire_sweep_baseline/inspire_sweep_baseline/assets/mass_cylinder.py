"""Bottom-heavy 1 kg cylinder with a bottom-centered body origin."""

from collections.abc import Callable

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


def bottom_heavy_mass_properties(radius, height, mass):
    """80% mass in bottom 20% height, 20% in the remaining upper cylinder.

    Both coaxial layers fill the same radius. Return COM height above the base
    and principal inertias about that COM, including parallel-axis terms.
    """
    lower_height, upper_height = 0.2 * height, 0.8 * height
    lower_mass, upper_mass = 0.8 * mass, 0.2 * mass
    lower_z, upper_z = 0.5 * lower_height, lower_height + 0.5 * upper_height
    com_z = (lower_mass * lower_z + upper_mass * upper_z) / mass
    transverse = sum(
        layer_mass * (3 * radius**2 + layer_height**2) / 12
        + layer_mass * (layer_z - com_z)**2
        for layer_mass, layer_height, layer_z in (
            (lower_mass, lower_height, lower_z), (upper_mass, upper_height, upper_z),
        )
    )
    axial = 0.5 * mass * radius**2
    return com_z, (transverse, transverse, axial)


@clone
def spawn_mass_cylinder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    # Explicitly call the standard spawner, rather than cfg.func (this wrapper).
    prim = sim_utils.spawn_cylinder(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs,
    )
    # Keep the object's pose at its base, matching the original cup's origin.
    # Otherwise Reach Z+9 cm would be measured from the cylinder's mid-height.
    mesh = prim.GetStage().GetPrimAtPath(f"{prim.GetPath()}/geometry/mesh")
    UsdGeom.Xformable(mesh).AddTranslateOp(opSuffix="bottom_origin").Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * cfg.height)
    )
    com_z, inertia = bottom_heavy_mass_properties(cfg.radius, cfg.height, cfg.mass_props.mass)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, com_z))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    return prim


@configclass
class MassCylinderCfg(sim_utils.CylinderCfg):
    func: Callable = spawn_mass_cylinder
    radius: float = 0.04
    height: float = 0.10
    axis: str = "Z"
    mass_props: sim_utils.MassPropertiesCfg = sim_utils.MassPropertiesCfg(mass=1.0)
