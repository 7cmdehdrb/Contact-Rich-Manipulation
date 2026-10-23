"""Clutter-specific observations, rewards, and failure conditions."""

from ..common.contact_sweep_clutter import all_object_tilt_angles

def any_object_fall_or_tilt(env, height_condition=1.04, fail_angle=0.9):
    """Apply the parent's drop/tilt rule to every real cylinder, then take any."""
    height = (
        env.scene["object_collection"].data.object_pos_w[..., 2]
        - env.scene.env_origins[:, None, 2]
    )
    return ((height < height_condition) | (all_object_tilt_angles(env) >= fail_angle)).any(dim=-1)
