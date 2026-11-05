"""Explicit population, speed and delayed direction curricula."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.contact_sweep_clutter_fusion import obstacle_population_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion_holding import reward_schedule
from ..curriculum_functions.wrench import goal_deadzone_schedule, feasibility_obstacle_population_schedule


@configclass
class WrenchCurriculumCfg:
    obstacle_population = CurriculumTermCfg(func=obstacle_population_schedule, params={
        'initial_gray': 3, 'initial_white': 3, 'final_gray': 4, 'final_white': 5,
        'gray_start_iteration': 12000, 'white_start_iteration': 10000, 'interval': 4000,
    })
    target_free_speed = CurriculumTermCfg(func=reward_schedule, params={
        'term_name': 'target_speed_penalty', 'parameter': 'free_speed',
        'points': ((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)), 'interval': 250,
    })
    target_goal_direction_deadzone = CurriculumTermCfg(func=goal_deadzone_schedule, params={
        'start_degrees': 30.0, 'start_iteration': 10000, 'interval': 3000, 'decrement_degrees': 5.0,
    })


@configclass
class WrenchV0_1CurriculumCfg(WrenchCurriculumCfg):
    target_goal_direction_deadzone = None


@configclass
class WrenchV0_2CurriculumCfg(WrenchCurriculumCfg):
    obstacle_population = CurriculumTermCfg(func=feasibility_obstacle_population_schedule)


@configclass
class WrenchV0_3CurriculumCfg(WrenchV0_2CurriculumCfg):
    obstacle_population = None
    target_free_speed = None
