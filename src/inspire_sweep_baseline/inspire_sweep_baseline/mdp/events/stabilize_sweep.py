from ..event_functions.stabilize_sweep import reset_scene
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from ..event_functions.wrench_refactor import initialize_physics
from ..event_functions.wrench_refactor_relative_ik_test import initialize_fast_geometry

@configclass
class StabilizeSweepEventsCfg:
    object_geometry = EventTerm(func=initialize_fast_geometry, mode='prestartup')
    object_physics = EventTerm(func=initialize_physics, mode='startup', params={'mass_scale': 1.5})
    reset_all = None
    object_spawn = EventTerm(func=reset_scene, mode='reset')
    robot_initial_pose = None
