"""Explicit impact schedules and alternating settling/impact refinement."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.curriculum_cfg import ContactSweepClutterFusionV1_0CurriculumCfg
from .curriculum import reward_schedule


@configclass
class FusionHoldingV0CurriculumCfg(ContactSweepClutterFusionV1_0CurriculumCfg):
    obstacle_population = ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.replace(
        params={**ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.params,
                'final_gray': 5, 'final_white': 5})
    # Same three speed schedules as the v1_4 discussion baseline.
    arm_free_speed = CurriculumTermCfg(func=reward_schedule, params={"term_name":'arm_speed_excess', "parameter":'free_speed',
        "points":((0, 1.0), (499, 1.0), (500, 0.98), (3300, 0.7)), "interval":250})
    eef_free_speed = CurriculumTermCfg(func=reward_schedule, params={"term_name":'eef_linear_speed', "parameter":'free_speed',
        "points":((0, 1.5), (499, 1.5), (500, 1.48), (13300, 0.2)), "interval":250})
    pushing_free_speed = CurriculumTermCfg(func=reward_schedule, params={"term_name":'pushing_speed', "parameter":'free_speed',
        "points":((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)), "interval":250})
    object_free_acceleration = CurriculumTermCfg(func=reward_schedule, params={"term_name":'object_acceleration', "parameter":'free_acceleration',
        "points":((0, 50.0), (3000, 50.0), (6000, 30.0), (10000, 30.0), (12000, 20.0)), "interval":250})
    wrench_free_force_change = CurriculumTermCfg(func=reward_schedule, params={"term_name":'wrench_change', "parameter":'free_force_change',
        "points":((0, 50.0), (3000, 50.0), (6000, 30.0), (10000, 30.0), (12000, 20.0)), "interval":250})
    wrench_free_moment_change = CurriculumTermCfg(func=reward_schedule, params={"term_name":'wrench_change', "parameter":'free_moment_change',
        "points":((0, 5.0), (3000, 5.0), (6000, 3.0), (10000, 3.0), (12000, 2.0)), "interval":250})
    obstacle_free_force = CurriculumTermCfg(func=reward_schedule, params={"term_name":'obstacle_contact', "parameter":'free_force',
        "points":((0, 80.0), (3000, 80.0), (6000, 50.0), (10000, 50.0), (12000, 30.0)), "interval":250})
    acceleration_weight = CurriculumTermCfg(func=reward_schedule, params={"term_name":'object_acceleration', "parameter":None,
        "points":((0, 0.0), (1000, 0.0), (3000, -0.01), (6000, -0.03), (10000, -0.03), (12000, -0.05)), "interval":250})
    wrench_weight = CurriculumTermCfg(func=reward_schedule, params={"term_name":'wrench_change', "parameter":None,
        "points":((0, 0.0), (1000, 0.0), (3000, -0.03), (6000, -0.1)), "interval":250})
    obstacle_weight = CurriculumTermCfg(func=reward_schedule, params={"term_name":'obstacle_contact', "parameter":None,
        "points":((0, 0.0), (1000, 0.0), (3000, -0.02), (6000, -0.05), (10000, -0.05), (12000, -0.1)), "interval":250})
    goal_position_scale = CurriculumTermCfg(func=reward_schedule, params={"term_name":'goal_hold', "parameter":'position_scale',
        "points":((0, 0.01), (1000, 0.02), (6000, 0.02), (10000, 0.01), (12000, 0.01), (16000, 0.005)), "interval":250})
    goal_speed_scale = CurriculumTermCfg(func=reward_schedule, params={"term_name":'goal_hold', "parameter":'speed_scale',
        "points":((0, 0.05), (6000, 0.05), (10000, 0.02)), "interval":250})
