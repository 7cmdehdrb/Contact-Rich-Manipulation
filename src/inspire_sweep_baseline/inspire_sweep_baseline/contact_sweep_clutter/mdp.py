"""Clutter-specific observations, rewards, and failure conditions."""

import math

from isaaclab.utils.math import matrix_from_quat


def all_object_tilt_angles(env):
    """Angle of every real cylinder's up axis relative to the live shelf normal."""
    objects = env.scene["object_collection"]
    object_up = matrix_from_quat(objects.data.object_quat_w)[..., :, 2]
    shelf_up = matrix_from_quat(env.scene["shelf"].data.root_quat_w)[..., :, 2]
    cosine = (object_up * shelf_up[:, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    return cosine.acos()


def all_object_tilt_cost(env, free_angle=math.radians(5.0), fail_angle=0.9):
    """Sum the parent's bounded per-object tilt cost over target and obstacles."""
    if not (0.0 <= free_angle < fail_angle <= math.pi):
        raise ValueError("Tilt thresholds must satisfy 0 <= free_angle < fail_angle <= pi")
    cost = ((all_object_tilt_angles(env) - free_angle) / (fail_angle - free_angle))
    return cost.clamp(0.0, 1.0).square().sum(dim=-1)


def any_object_fall_or_tilt(env, height_condition=1.04, fail_angle=0.9):
    """Apply the parent's drop/tilt rule to every real cylinder, then take any."""
    height = (
        env.scene["object_collection"].data.object_pos_w[..., 2]
        - env.scene.env_origins[:, None, 2]
    )
    return ((height < height_condition) | (all_object_tilt_angles(env) >= fail_angle)).any(dim=-1)


def initial_obstacle_positions(env):
    """Return the reset-time 10 x [present, x_root, y_root, z_root] snapshot.

    The scene reset owns this cache. It excludes the target and zero-pads absent
    obstacle slots. Reading observations never refreshes positions from physics.
    """
    return env._initial_obstacle_observation
