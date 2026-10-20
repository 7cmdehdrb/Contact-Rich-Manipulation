"""User-specified schedules, in Learning iterations (not environment steps)."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from .mdp import reward_schedule


@configclass
class ContactSweepClutterV1CurriculumCfg:
    pushing_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "pushing_speed", "parameter": "free_speed",
            "start": 0.1, "decrement": 0.01, "interval": 3000, "minimum": 0.05,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_speed_excess", "parameter": "free_speed",
            "start": 1.0, "decrement": 0.05, "interval": 5000, "minimum": 0.8,
        },
    )
    arm_action_rate_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_action_rate",
            "start": -0.03, "decrement": 0.005, "interval": 3000, "minimum": -0.06,
        },
    )
    arm_joint_velocity_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_joint_velocity",
            "start": -0.03, "decrement": 0.005, "interval": 3000, "minimum": -0.06,
        },
    )
