"""Fusion v3 schedules in the runner's zero-based Learning iteration units."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.curriculum import goal_deadzone_schedule
from ..contact_sweep_clutter_fusion.curriculum_cfg import (
    ContactSweepClutterFusionCurriculumCfg,
)
from .curriculum import delayed_reward_schedule


@configclass
class ContactSweepClutterFusionV3CurriculumCfg(ContactSweepClutterFusionCurriculumCfg):
    """
    Free Speed Curriculum
    """

    # Pushing Free Speed : 0.1 -> 0.03
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

    # Arm Free Speed : 1.0 -> 0.7
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

    # Action Rate Weight : -0.03 -> -0.06
    action_rate_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "action_rate",
            "start": -0.03,
            "step": -0.005,
            "final": -0.06,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    # EEF Free Speed : 1.5 -> 0.3
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

    """
    Deadzone Curriculum
    """

    # Goal Deadzone : 60 -> 0
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 60.0,
            "hold_iterations": 1000,
            "interval": 100,
            "decrement_degrees": 1.0,
        },
    )

    """
    Contact Curriculum
    """

    # Contact Any Weight : 0.4 -> 2.0
    contact_any_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "contact_any",
            "start": 0.4,
            "step": 0.02,
            "final": 2.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Contact Count Weight : 0.4 -> 2.0
    contact_count_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "contact_count",
            "start": 0.4,
            "step": 0.02,
            "final": 2.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    """
    Wrench Curriculum
    """

    # Wrench Free Force Rate : 2000 -> 1000
    wrench_free_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_force_rate",
            "start": 2000.0,
            "step": -50.0,
            "final": 1000.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Free Moment Rate : 100 -> 50
    wrench_free_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_moment_rate",
            "start": 100.0,
            "step": -2.5,
            "final": 50.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Saturation Force Rate : 5000 -> 2500
    wrench_saturation_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "saturation_force_rate",
            "start": 5000.0,
            "step": -125.0,
            "final": 2500.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Saturation Moment Rate : 250 -> 125.0
    wrench_saturation_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "saturation_moment_rate",
            "start": 250.0,
            "step": -6.25,
            "final": 125.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
