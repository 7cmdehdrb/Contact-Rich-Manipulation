"""Holding smoothness schedules, fixed radius, IK direction and arm cutoff."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.contact_sweep_clutter_fusion_holding import reward_schedule
from ..curriculum_functions.wrench import goal_deadzone_schedule
from ..curriculum_functions.wrench_refactor import arm_velocity_threshold_schedule

@configclass
class WrenchRefactorRelativeIKV0_2TestCurriculumCfg:
    eef_free_speed = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'eef_linear_speed', 'parameter': 'free_speed', 'points': ((0, 1.5), (499, 1.5), (500, 1.48), (13300, 0.2)), 'interval': 250})
    pushing_free_speed = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'pushing_speed', 'parameter': 'free_speed', 'points': ((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)), 'interval': 250})
    object_free_acceleration = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'object_acceleration', 'parameter': 'free_acceleration', 'points': ((0, 50.0), (3000, 50.0), (6000, 30.0), (10000, 30.0), (12000, 20.0)), 'interval': 250})
    wrench_free_force_change = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': 'free_force_change', 'points': ((0, 50.0), (3000, 50.0), (6000, 30.0), (10000, 30.0), (12000, 20.0)), 'interval': 250})
    wrench_free_moment_change = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': 'free_moment_change', 'points': ((0, 5.0), (3000, 5.0), (6000, 3.0), (10000, 3.0), (12000, 2.0)), 'interval': 250})
    obstacle_free_force = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'obstacle_contact', 'parameter': 'free_force', 'points': ((0, 80.0), (3000, 80.0), (6000, 50.0), (10000, 50.0), (12000, 30.0)), 'interval': 250})
    acceleration_weight = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'object_acceleration', 'parameter': None, 'points': ((0, 0.0), (1000, 0.0), (3000, -0.01), (6000, -0.03), (10000, -0.03), (12000, -0.05)), 'interval': 250})
    wrench_weight = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'wrench_change', 'parameter': None, 'points': ((0, 0.0), (1000, 0.0), (3000, -0.03), (6000, -0.1)), 'interval': 250})
    obstacle_weight = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'obstacle_contact', 'parameter': None, 'points': ((0, 0.0), (1000, 0.0), (3000, -0.02), (6000, -0.05), (10000, -0.05), (12000, -0.1)), 'interval': 250})
    goal_speed_scale = CurriculumTermCfg(func=reward_schedule, params={'term_name': 'goal_hold', 'parameter': 'speed_scale', 'points': ((0, 0.05), (6000, 0.05), (10000, 0.02)), 'interval': 250})
    target_goal_direction_deadzone = CurriculumTermCfg(func=goal_deadzone_schedule, params={'start_degrees': 30.0, 'start_iteration': 10000, 'interval': 3000, 'decrement_degrees': 5.0})
    arm_joint_velocity_threshold = CurriculumTermCfg(func=arm_velocity_threshold_schedule, params={'initial': 4.0, 'minimum': 2.0, 'start_iteration': 10000, 'interval': 250, 'decrement': 0.05})


@configclass
class WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg(WrenchRefactorRelativeIKV0_2TestCurriculumCfg):
    eef_free_speed = None
    pushing_free_speed = None
