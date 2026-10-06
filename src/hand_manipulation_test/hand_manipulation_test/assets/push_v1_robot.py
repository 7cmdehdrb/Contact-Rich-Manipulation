"""V1-only exposure of palmar thumb pads otherwise occluded by their carriers."""

from pxr import Gf, UsdGeom

from isaaclab.sim.utils import clone, get_current_stage

from .contact_robot import spawn_contact_robot


THUMB_PAD_EXPOSURE_M = 0.012
EXPOSED_THUMB_PAD_NAMES = ("inspire_thumb_force_sensor_3", "inspire_thumb_force_sensor_4")


@clone
def spawn_push_v1_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep bodies, joints, mass and visuals; expose two existing collision pads.

    The reset-time hull audit finds up to 9.99 mm of carrier occlusion,
    including its reserve. A 12 mm displacement in the authored H +Y
    direction leaves the actual filtered palmar sensors in front of it.
    Reaching, Contact and Push-v0 continue using their original spawners.
    """

    root = spawn_contact_robot.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    stage = get_current_stage()
    cache = UsdGeom.XformCache()
    root_path = str(root.GetPath())
    hand = stage.GetPrimAtPath(root_path + "/inspire_base_link")
    displacement = cache.GetLocalToWorldTransform(hand).TransformDir(Gf.Vec3d(0., THUMB_PAD_EXPOSURE_M, 0.))
    for name in EXPOSED_THUMB_PAD_NAMES:
        collision = stage.GetPrimAtPath(root_path + "/" + name + "/collisions")
        if not collision.IsValid():
            raise RuntimeError(f"Missing palmar thumb collider: {collision.GetPath()}")
        world = cache.GetLocalToWorldTransform(collision)
        world.SetTranslateOnly(world.ExtractTranslation() + displacement)
        local = world * cache.GetLocalToWorldTransform(collision.GetParent()).GetInverse()
        transform = UsdGeom.Xformable(collision)
        transform.ClearXformOpOrder()
        transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local)
    return root


__all__ = ["spawn_push_v1_robot", "THUMB_PAD_EXPOSURE_M", "EXPOSED_THUMB_PAD_NAMES"]
