"""Simple family: pinned effective v0_2_2_4 configuration, without historical Cfg inheritance."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.contact_sweep_clutter_fusion_holding import reward_schedule
from ..curriculum_functions.wrench import goal_deadzone_schedule
from ..curriculum_functions.wrench_refactor import arm_velocity_threshold_schedule

@configclass
class SweepSimpleCurriculumCfg:
    eef_free_speed = None
    pushing_free_speed = None
    object_free_acceleration = None
    wrench_free_force_change = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': 'free_force_change', 'points': ((0, 50.0), (3000, 50.0), (6000, 30.0), (10000, 30.0), (12000, 20.0)), 'interval': 250})
    wrench_free_moment_change = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': 'free_moment_change', 'points': ((0, 5.0), (3000, 5.0), (6000, 3.0), (10000, 3.0), (12000, 2.0)), 'interval': 250})
    obstacle_free_force = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'obstacle_contact', 'parameter': 'free_force', 'points': ((0, 80.0), (3000, 80.0), (6000, 50.0), (10000, 50.0), (12000, 30.0)), 'interval': 250})
    acceleration_weight = None
    wrench_weight = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': None, 'points': ((0, 0.0), (1000, 0.0), (3000, -0.03), (6000, -0.1)), 'interval': 250})
    obstacle_weight = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'obstacle_contact', 'parameter': None, 'points': ((0, 0.0), (1000, 0.0), (3000, -0.02), (6000, -0.05), (10000, -0.05), (12000, -0.1)), 'interval': 250})
    goal_speed_scale = None
    target_goal_direction_deadzone = CurriculumTermCfg(func=goal_deadzone_schedule, params={'start_degrees': 30.0, 'start_iteration': 10000, 'interval': 3000, 'decrement_degrees': 5.0})
    arm_joint_velocity_threshold = CurriculumTermCfg(func=arm_velocity_threshold_schedule, params={'initial': 4.0, 'minimum': 2.0, 'start_iteration': 10000, 'interval': 250, 'decrement': 0.05})


@configclass
class SweepSimpleHardCurriculumCfg(SweepSimpleCurriculumCfg):
    target_goal_direction_deadzone = None


@configclass
class SweepSimpleFingerV0TestCurriculumCfg(SweepSimpleCurriculumCfg):
    arm_joint_velocity_threshold = None


@configclass
class SweepSimpleFingerV1CurriculumCfg(SweepSimpleCurriculumCfg):
    arm_joint_velocity_threshold = None
