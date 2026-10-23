"""Clutter-specific observations, rewards, and failure conditions."""

from isaaclab.utils.math import matrix_from_quat

def all_object_tilt_angles(env):
    """Angle of every real cylinder's up axis relative to the live shelf normal."""
    objects = env.scene["object_collection"]
    object_up = matrix_from_quat(objects.data.object_quat_w)[..., :, 2]
    shelf_up = matrix_from_quat(env.scene["shelf"].data.root_quat_w)[..., :, 2]
    cosine = (object_up * shelf_up[:, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    return cosine.acos()
