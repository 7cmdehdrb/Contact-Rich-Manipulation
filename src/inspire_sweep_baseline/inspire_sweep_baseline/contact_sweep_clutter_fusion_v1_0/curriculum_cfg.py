"""Absolute iteration schedules; no inherited terms or tactile schedules."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.curriculum import goal_deadzone_schedule
from ..contact_sweep_clutter_fusion_v3.curriculum import delayed_reward_schedule
from .population import obstacle_population_schedule


@configclass
class ContactSweepClutterFusionV1_0CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={"initial_gray": 1, "initial_white": 1, "final_gray": 3, "final_white": 3,
                "gray_start_iteration": 4500, "white_start_iteration": 3000, "interval": 3000},
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "pushing_speed", "parameter": "free_speed",
                "start": 0.1, "step": -0.01, "final": 0.03,
                "start_iteration": 500, "interval": 200},
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "arm_speed_excess", "parameter": "free_speed",
                "start": 1.0, "step": -0.02, "final": 0.7,
                "start_iteration": 500, "interval": 200},
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "eef_linear_speed", "parameter": "free_speed",
                "start": 1.5, "step": -0.02, "final": 0.3,
                "start_iteration": 500, "interval": 200},
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={"start_degrees": 30.0, "hold_iterations": 3000,
                "interval": 500, "decrement_degrees": 5.0},
    )
    goal_hold_tolerance = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "goal_hold", "parameter": "position_tolerance",
                "start": 0.03, "step": -0.01, "final": 0.01,
                "start_iteration": 1000, "interval": 500},
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "free_force_change",
                "start": 50.0, "step": -1.0, "final": 10.0,
                "start_iteration": 1000, "interval": 250},
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "free_moment_change",
                "start": 5.0, "step": -0.1, "final": 1.0,
                "start_iteration": 1000, "interval": 250},
    )
    object_free_acceleration = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "object_acceleration", "parameter": "free_acceleration",
                "start": 1.0, "step": -0.02, "final": 0.1,
                "start_iteration": 1000, "interval": 250},
    )
