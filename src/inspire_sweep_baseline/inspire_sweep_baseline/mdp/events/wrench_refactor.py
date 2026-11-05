"""Independent Refactor lifecycle: no physics randomization on reset."""
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from ..event_functions.wrench_refactor import initialize_geometry, initialize_physics, reset_scene


@configclass
class WrenchRefactorEventsCfg:
    object_geometry = EventTerm(func=initialize_geometry, mode='prestartup')
    object_physics = EventTerm(func=initialize_physics, mode='startup', params={'mass_scale': 1.5})
    reset_all = EventTerm(func=reset_scene_to_default, mode='reset')
    object_spawn = EventTerm(func=reset_scene, mode='reset', params={
        'spawn_radius': .02, 'edge_clearance': .02,
        'excluded_x_half_angle': 0.5235987755982988, 'observation_noise': .01,
        'visibility_threshold': .35, 'visibility_flip_probability': .01,
        'observation_swap_probability': .1,
    })
