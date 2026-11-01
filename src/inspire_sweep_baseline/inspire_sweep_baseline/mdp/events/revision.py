"""Independent Revision event configuration."""

from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..event_functions.revision import initialize_revision_geometry, reset_revision_scene
from ..event_functions.contact_sweep_clutter_physics import initialize_clutter_materials
from ..event_functions.contact_sweep_clutter_fusion_holding_osc_object_random import randomize_prop_physics
from ..common.contact_sweep_clutter_fusion import OBJECT_MASS_SCALE


@configclass
class RevisionEventsCfg:
    object_geometry = EventTerm(func=initialize_revision_geometry, mode='prestartup')
    reset_all = EventTerm(func=reset_scene_to_default, mode='reset')
    object_spawn = EventTerm(func=reset_revision_scene, mode='reset', params={
        'gray_obstacles': 1, 'white_obstacles': 1, 'spawn_radius': .02,
        'goal_distance': .15, 'edge_clearance': .02,
        'excluded_x_half_angle': 0.5235987755982988, 'observation_noise': .01,
        'visibility_threshold': .35, 'visibility_flip_probability': .01,
        'observation_swap_probability': .1,
    })
    object_materials = EventTerm(func=initialize_clutter_materials, mode='startup')
    object_physics_randomization = EventTerm(func=randomize_prop_physics, mode='reset',
                                            params={'mass_scale': OBJECT_MASS_SCALE})
