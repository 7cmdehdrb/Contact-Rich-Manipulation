"""Revision failure predicates and inclusive timeout partitions."""

import torch
from isaaclab.envs.mdp import time_out

from ..common.contact_sweep_clutter import all_object_tilt_angles
from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..common.contact_sweep_clutter_fusion_object_state import active_object_mask
from ..common.revision_goal import goal_radius, object_geometry_in_shelf
from .goal_terminations import _object_position


def any_active_object_fall_or_tilt(env, bottom_clearance=0.01, fail_angle=0.9):
    """Evaluate actual bounds bottoms, rather than shape-dependent root heights."""
    position = env.scene["object_collection"].data.object_pos_w
    _, bottom = object_geometry_in_shelf(env)
    failed = (
        ~torch.isfinite(bottom)
        | (bottom < SHELF_SURFACE_HEIGHT - bottom_clearance)
        | (all_object_tilt_angles(env) >= fail_angle)
    )
    return (failed & active_object_mask(env, position)).any(dim=-1)


def timeout_goal_reached(env):
    """Success is finite XY distance <= 3 cm, without speed or hold-time gates."""
    goal = env.command_manager.get_command("target_goal_pos")
    distance = torch.linalg.vector_norm(_object_position(env)[:, :2] - goal[:, :2], dim=-1)
    return torch.isfinite(distance) & (distance <= goal_radius(env))


def success_at_timeout(env):
    env.record_goal_accuracy_metrics()
    return time_out(env) & timeout_goal_reached(env)


def unsuccessful_timeout(env):
    return time_out(env) & ~timeout_goal_reached(env)
