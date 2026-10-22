"""Absolute iteration schedules; no inherited terms or tactile schedules."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.curriculum import goal_deadzone_schedule
from ..contact_sweep_clutter_fusion_v3.curriculum import delayed_reward_schedule
from ..contact_sweep_clutter_fusion_v1_0.population import obstacle_population_schedule
from ..contact_sweep_clutter_fusion_v1_1.curriculum import staged_reward_threshold


@configclass
class ContactSweepClutterFusionV1_1_1CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "initial": 100.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
