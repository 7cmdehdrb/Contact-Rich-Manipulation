"""Five absolute schedules in the runner's Learning iteration units."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1_curriculum.mdp import reward_schedule
from .curriculum import goal_deadzone_schedule


@configclass
class ContactSweepClutterFusionCurriculumCfg:
    pushing_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "decrement": 0.01,
            "interval": 300,
            "minimum": 0.05,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "decrement": 0.02,
            "interval": 300,
            "minimum": 0.8,
        },
    )
    action_rate_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "action_rate",
            "start": -0.03,
            "decrement": 0.005,
            "interval": 300,
            "minimum": -0.06,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "decrement": 0.02,
            "interval": 300,
            "minimum": 0.7,
        },
    )
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 60.0,
            "hold_iterations": 500,
            "interval": 200,
            "decrement_degrees": 1.0,
        },
    )
