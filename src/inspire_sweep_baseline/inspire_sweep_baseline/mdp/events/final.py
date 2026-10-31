"""Final v0 reset and randomized prop physics."""

from ..common.contact_sweep_clutter_fusion import OBJECT_MASS_SCALE
from ..common.contact_sweep_clutter_fusion import SHELF_HALF_EXTENTS
from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..event_functions.contact_sweep_clutter_physics import initialize_clutter_materials
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import math as math
from ..event_functions.contact_sweep_clutter_fusion_holding_osc_object_random import randomize_large_prop_scene, randomize_prop_physics


@configclass
class FinalEventsCfg:
    reset_all = EventTerm(func=reset_scene_to_default, mode='reset')
    object_spawn = EventTerm(
        func=randomize_large_prop_scene,
        mode='reset',
        params={
            'spawn_radius': 0.02,
            'goal_distance': 0.15,
            'object_radius': 0.04,
            'edge_clearance': 0.02,
            'shelf_half_extents': SHELF_HALF_EXTENTS,
            'shelf_surface_height': SHELF_SURFACE_HEIGHT,
            'excluded_x_half_angle': math.radians(30.0),
            'observation_noise': 0.01,
            'visibility_threshold': 0.35,
            'visibility_flip_probability': 0.01,
            'observation_swap_probability': 0.1,
            'gray_obstacles': 1,
            'white_obstacles': 1,
        },
    )
    physics_materials = EventTerm(func=initialize_clutter_materials, mode='startup')
    physics_randomization = EventTerm(func=randomize_prop_physics, mode='reset', params={'mass_scale': OBJECT_MASS_SCALE})


from ..event_functions.final import randomize_object_sizes, randomize_final_scene


@configclass
class FinalV1EventsCfg(FinalEventsCfg):
    object_size_randomization = EventTerm(func=randomize_object_sizes, mode='prestartup',
                                          params={'scale_range': (.9, 1.1)})
    object_spawn = EventTerm(func=randomize_final_scene, mode='reset', params=FinalEventsCfg().object_spawn.params)
    physics_materials = None
    physics_randomization = None
    object_materials = FinalEventsCfg().physics_materials
    object_physics_randomization = FinalEventsCfg().physics_randomization


@configclass
class FinalV2EventsCfg(FinalV1EventsCfg):
    object_spawn = EventTerm(func=randomize_final_scene, mode='reset',
                            params={**FinalV1EventsCfg().object_spawn.params, 'wide_target': True})


from ..event_functions.final import randomize_final_target_scene


@configclass
class FinalGoalAccuracyV0_2_1EventsCfg(FinalV1EventsCfg):
    object_spawn = FinalV1EventsCfg().object_spawn.replace(func=randomize_final_target_scene)
