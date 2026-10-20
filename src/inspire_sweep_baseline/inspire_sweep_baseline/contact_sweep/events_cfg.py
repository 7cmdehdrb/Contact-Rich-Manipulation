"""Explicit reset order and source-material startup for Contact Sweep."""

import math
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..mdp.goal_events import randomize_goal_object_scene
from ..mdp.realistic_physics import initialize_realistic_materials, randomize_scaled_episode_physics
from .common import OBJECT_MASS_SCALE, OBJECT_RADIUS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT


@configclass
class ContactSweepEventsCfg:
    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(
        func=randomize_goal_object_scene, mode="reset",
        params={
            "spawn_radius": 0.02, "goal_distance": 0.15, "object_radius": OBJECT_RADIUS,
            "edge_clearance": 0.02, "shelf_half_extents": SHELF_HALF_EXTENTS,
            "shelf_surface_height": SHELF_SURFACE_HEIGHT,
            "excluded_x_half_angle": math.radians(15.0),
        },
    )
    physics_materials = EventTerm(func=initialize_realistic_materials, mode="startup")
    physics_randomization = EventTerm(
        func=randomize_scaled_episode_physics, mode="reset", params={"mass_scale": OBJECT_MASS_SCALE},
    )
