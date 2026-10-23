"""Self-contained baseline configuration, derived from the v2-mass behavior.

Inherited baseline experiment varying only object mass and contact friction."""

from ..event_functions.realistic_physics import initialize_realistic_materials
from ..event_functions.realistic_physics import randomize_episode_physics
from ..event_functions.sweep import randomize_single_object_scene
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

@configclass
class EventsCfg:
    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(
        func=randomize_single_object_scene, mode="reset",
        params={"sweep_direction": "both", "object_radius": 0.04,
                "position_noise": 0.02, "initial_position": (-0.75, 0.0, 1.05)},
    )


@configclass
class RealisticPhysicsEventsCfg(EventsCfg):
    # Source spawners already bind Average materials before physics cloning.
    # Cache Tensor API buffers and bounded Gaussian material pools once.
    physics_materials = EventTerm(func=initialize_realistic_materials, mode="startup")
    physics_randomization = EventTerm(func=randomize_episode_physics, mode="reset")
