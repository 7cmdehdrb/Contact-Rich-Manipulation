"""Independent reset configuration with initial-only visibility classification.

Explicit lifecycle events for the current episode's obstacle population."""

from ..common.contact_sweep_clutter_fusion import OBJECT_MASS_SCALE
from ..common.contact_sweep_clutter_fusion import OBJECT_RADIUS
from ..common.contact_sweep_clutter_fusion import SHELF_HALF_EXTENTS
from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..event_functions.contact_sweep_clutter_fusion import randomize_population_clutter_scene
from ..event_functions.contact_sweep_clutter_fusion import randomize_visible_clutter_scene
from ..event_functions.contact_sweep_clutter_physics import initialize_clutter_materials
from ..event_functions.contact_sweep_clutter_physics import randomize_clutter_episode_physics
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import math as math

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


@configclass
class ContactSweepClutterFusionV1_0EventsCfg:
    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(
        func=randomize_population_clutter_scene, mode="reset",
        params={
            "spawn_radius": 0.02, "goal_distance": 0.15, "object_radius": OBJECT_RADIUS,
            "edge_clearance": 0.02, "shelf_half_extents": SHELF_HALF_EXTENTS,
            "shelf_surface_height": SHELF_SURFACE_HEIGHT,
            "excluded_x_half_angle": math.radians(30.0), "observation_noise": 0.01,
            "visibility_threshold": 0.35, "visibility_flip_probability": 0.01,
            "observation_swap_probability": 0.1,
            "gray_obstacles": 1, "white_obstacles": 1,
        },
    )
    physics_materials = EventTerm(func=initialize_clutter_materials, mode="startup")
    physics_randomization = EventTerm(
        func=randomize_clutter_episode_physics, mode="reset", params={"mass_scale": OBJECT_MASS_SCALE},
    )
