"""Independent Refactor lifecycle: no physics randomization on reset."""
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from ..event_functions.wrench_refactor import initialize_geometry, initialize_physics, reset_scene, reset_robot_from_goal_ik


@configclass
class WrenchRefactorRelativeIKV0_2TestEventsCfg:
    object_geometry = EventTerm(func=initialize_geometry, mode='prestartup')
    object_physics = EventTerm(func=initialize_physics, mode='startup', params={'mass_scale': 1.5})
    reset_all = EventTerm(func=reset_scene_to_default, mode='reset')
    object_spawn = EventTerm(func=reset_scene, mode='reset', params={
        'spawn_radius': .02, 'edge_clearance': .02, 'min_goal_distance': .06, 'max_goal_distance': .12,
        'excluded_x_half_angle': 0.5235987755982988, 'observation_noise': .01,
        'visibility_threshold': .35, 'visibility_flip_probability': .01,
        'observation_swap_probability': .1,
    })
    robot_initial_pose = EventTerm(func=reset_robot_from_goal_ik, mode='reset', params={
        'front_clearance': (0.08, 0.10), 'lateral_offset': (0.10, 0.18),
        'height_jitter': 0.02, 'height_offset': 0.01,
    })


from ..event_functions.wrench_refactor_relative_ik_test import grouped_reset_scene, randomize_reset_physics


@configclass
class WrenchRefactorRelativeIKV0_2_1ResetTestEventsCfg(WrenchRefactorRelativeIKV0_2TestEventsCfg):
    # Startup prepares immutable geometry/material buffers. Reset updates values.
    reset_physics = EventTerm(func=randomize_reset_physics, mode='reset', params={'mass_scale': 1.5})

    def __post_init__(self):
        self.object_spawn.func = grouped_reset_scene
        self.object_spawn.params['ik_params'] = dict(self.robot_initial_pose.params)
        self.robot_initial_pose = None
