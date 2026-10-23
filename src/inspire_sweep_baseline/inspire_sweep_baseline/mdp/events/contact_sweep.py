"""Explicit reset order and source-material startup for Contact Sweep."""

from ..common.contact_sweep import OBJECT_MASS_SCALE
from ..common.contact_sweep import OBJECT_RADIUS
from ..common.contact_sweep import SHELF_HALF_EXTENTS
from ..common.contact_sweep import SHELF_SURFACE_HEIGHT
from ..event_functions.goal_events import randomize_goal_object_scene
from ..event_functions.realistic_physics import initialize_realistic_materials
from ..event_functions.realistic_physics import randomize_scaled_episode_physics
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import math as math

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
