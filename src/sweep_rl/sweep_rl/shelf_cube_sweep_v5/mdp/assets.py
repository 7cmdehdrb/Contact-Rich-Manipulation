"""Asset spawners owned by the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Gf, UsdPhysics

from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_bottom_heavy_cuboid(
    prim_path: str,
    cfg: "BottomHeavyCuboidCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    prim = spawn_cuboid.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    mass_api = UsdPhysics.MassAPI(prim)
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*cfg.center_of_mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*cfg.diagonal_inertia))
    w, x, y, z = cfg.principal_axes
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return prim


@configclass
class BottomHeavyCuboidCfg(CuboidCfg):
    func = spawn_bottom_heavy_cuboid
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    diagonal_inertia: tuple[float, float, float] = (1.0, 1.0, 1.0)
    principal_axes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
