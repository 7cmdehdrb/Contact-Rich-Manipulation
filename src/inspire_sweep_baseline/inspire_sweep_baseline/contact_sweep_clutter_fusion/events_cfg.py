"""Independent reset configuration with initial-only visibility classification."""

import math

from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..contact_sweep_clutter.physics import initialize_clutter_materials, randomize_clutter_episode_physics
from .common import OBJECT_MASS_SCALE, OBJECT_RADIUS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT
from .events import randomize_visible_clutter_scene


@configclass
class ContactSweepClutterFusionEventsCfg:
    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(
        func=randomize_visible_clutter_scene, mode="reset",
        params={
            "spawn_radius": 0.02, "goal_distance": 0.15, "object_radius": OBJECT_RADIUS,
            "edge_clearance": 0.02, "shelf_half_extents": SHELF_HALF_EXTENTS,
            "shelf_surface_height": SHELF_SURFACE_HEIGHT,
            "excluded_x_half_angle": math.pi / 3.0, "observation_noise": 0.01,
            "visibility_threshold": 0.35, "visibility_flip_probability": 0.01,
        },
    )
    physics_materials = EventTerm(func=initialize_clutter_materials, mode="startup")
    physics_randomization = EventTerm(
        func=randomize_clutter_episode_physics, mode="reset", params={"mass_scale": OBJECT_MASS_SCALE},
    )
