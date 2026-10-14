"""Variant-local exposure of the physically occluded central palm collider."""

from collections.abc import Callable

from pxr import Gf, UsdGeom

from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .robot import Ur5eAxia80InspireSpawnerCfg, make_robot_cfg, spawn_ur5e_axia80_inspire


CENTRAL_PALM_COLLIDER_OFFSET_H_M = (0.0, 0.035, 0.0)


@clone
def spawn_contact_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Move only the central pad's collider, before scene physics replication.

    The original base hull and thumb carrier shield the source pad from a
    6 cm Cube. A 35 mm outward shift was checked with live PhysX contact probes
    at hand openness .90/.94/.98. Body, joint, visual, mass, and the other
    sixteen palmar pad geometries remain the package-local source model.
    """

    # Apply the override to the source before the outer clone copies it. The
    # base decorator uses functools.wraps, so __wrapped__ is its single spawn.
    robot_root = spawn_ur5e_axia80_inspire.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    # The outer @clone resolves the regex path to this real source prim.
    root_path = str(robot_root.GetPath())
    stage = get_current_stage()
    hand = stage.GetPrimAtPath(root_path + "/inspire_base_link")
    collision = stage.GetPrimAtPath(root_path + "/inspire_palm_force_sensor/collisions")
    if not hand.IsValid() or not collision.IsValid():
        raise RuntimeError("The contact variant requires the original central palm collision subtree")
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(collision)
    displacement = cache.GetLocalToWorldTransform(hand).TransformDir(Gf.Vec3d(*CENTRAL_PALM_COLLIDER_OFFSET_H_M))
    world.SetTranslateOnly(world.ExtractTranslation() + displacement)
    local = world * cache.GetLocalToWorldTransform(collision.GetParent()).GetInverse()
    xform = UsdGeom.Xformable(collision)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local)
    return robot_root


@configclass
class ContactRobotSpawnerCfg(Ur5eAxia80InspireSpawnerCfg):
    func: Callable = spawn_contact_robot


def make_contact_robot_cfg():
    """Return an independent robot config using only the contact variant spawner."""

    cfg = make_robot_cfg()
    cfg.spawn = ContactRobotSpawnerCfg()
    return cfg


__all__ = ["CENTRAL_PALM_COLLIDER_OFFSET_H_M", "make_contact_robot_cfg"]
