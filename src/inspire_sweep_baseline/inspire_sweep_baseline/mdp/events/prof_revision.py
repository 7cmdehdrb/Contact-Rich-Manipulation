"""Simple startup physics and integrated circle/Warp reset."""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from ..event_functions.wrench_refactor import initialize_physics
from ..event_functions.wrench_refactor_relative_ik_test import initialize_fast_geometry, fast_reset_scene
from ..event_functions.prof_revision import reset_scene
from ..event_functions.prof_revision import reset_hard_scene
from ..common.prof_revision_layout import ProfRevisionHardLayout

@configclass
class ProfRevisionEventsCfg:
    object_geometry = EventTerm(func=initialize_fast_geometry, mode='prestartup')
    object_physics = EventTerm(func=initialize_physics, mode='startup', params={'mass_scale': 1.5})
    reset_all = None
    object_spawn = EventTerm(func=reset_scene, mode='reset', params={'spawn_radius': 0.02, 'edge_clearance': 0.02, 'min_goal_distance': 0.06, 'max_goal_distance': 0.12, 'excluded_x_half_angle': 0.5235987755982988, 'observation_noise': 0.01, 'visibility_threshold': 0.35, 'visibility_flip_probability': 0.01, 'observation_swap_probability': 0.1, 'ik_params': dict(front_clearance=(0.08, 0.1), lateral_offset=(0.1, 0.18), height_jitter=0.02, height_offset=0.01)})
    robot_initial_pose = None


@configclass
class ProfRevisionHardEventsCfg(ProfRevisionEventsCfg):
    def __post_init__(self):
        self.object_geometry.params['layout_class'] = ProfRevisionHardLayout
        self.object_spawn.func = reset_hard_scene
        self.object_spawn.params.update(min_goal_distance=.05, max_goal_distance=None)


@configclass
class ProfRevisionV0_3_3IKEventsCfg(ProfRevisionEventsCfg):
    def __post_init__(self):
        self.object_spawn.params['ik_params'].update(initial_pose_mode='front_area', front_y_bounds=(-.5, .5))
