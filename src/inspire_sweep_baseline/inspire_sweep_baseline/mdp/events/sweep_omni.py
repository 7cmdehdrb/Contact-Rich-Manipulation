"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards."""

from ..common.sweep_omni import OBJECT_RADIUS
from ..common.sweep_omni import SHELF_HALF_EXTENTS
from ..common.sweep_omni import SHELF_SURFACE_HEIGHT
from ..event_functions.goal_events import randomize_goal_object_scene
from .sweep import RealisticPhysicsEventsCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

@configclass
class OmniEventsCfg(RealisticPhysicsEventsCfg):
    object_spawn = EventTerm(
        func=randomize_goal_object_scene,
        mode="reset",
        params={
            "spawn_radius": 0.02,
            "goal_distance": 0.15,
            "object_radius": OBJECT_RADIUS,
            "edge_clearance": 0.02,
            "shelf_half_extents": SHELF_HALF_EXTENTS,
            "shelf_surface_height": SHELF_SURFACE_HEIGHT,
        },
    )
