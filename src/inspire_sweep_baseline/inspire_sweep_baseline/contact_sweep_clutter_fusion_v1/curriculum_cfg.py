"""Fusion v1 schedules in the runner's zero-based Learning iteration units."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.curriculum import goal_deadzone_schedule
from ..contact_sweep_clutter_fusion.curriculum_cfg import ContactSweepClutterFusionCurriculumCfg
from .curriculum import delayed_reward_schedule


@configclass
class ContactSweepClutterFusionV1CurriculumCfg(ContactSweepClutterFusionCurriculumCfg):
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "pushing_speed", "parameter": "free_speed",
                "start": 0.1, "step": -0.01, "final": 0.03},
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "arm_speed_excess", "parameter": "free_speed",
                "start": 1.0, "step": -0.02, "final": 0.8},
    )
    action_rate_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "action_rate", "start": -0.03,
                "step": -0.005, "final": -0.06},
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "eef_linear_speed", "parameter": "free_speed",
                "start": 1.5, "step": -0.02, "interval": 500, "final": 0.5},
    )
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={"start_degrees": 60.0, "hold_iterations": 1000,
                "interval": 100, "decrement_degrees": 1.0},
    )
    contact_any_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "contact_any", "start": 0.4, "step": 0.02, "final": 2.0},
    )
    contact_count_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "contact_count", "start": 0.4, "step": 0.02, "final": 2.0},
    )
    wrench_free_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "free_force_rate",
                "start": 2000.0, "step": -50.0, "final": 500.0},
    )
    wrench_free_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "free_moment_rate",
                "start": 100.0, "step": -2.5, "final": 25.0},
    )
    wrench_saturation_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "saturation_force_rate",
                "start": 5000.0, "step": -125.0, "final": 1250.0},
    )
    wrench_saturation_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={"term_name": "wrench_change", "parameter": "saturation_moment_rate",
                "start": 250.0, "step": -6.25, "final": 62.5},
    )
