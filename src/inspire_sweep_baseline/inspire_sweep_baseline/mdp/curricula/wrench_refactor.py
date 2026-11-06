"""Refactor direction schedule and Relative arm velocity termination schedule."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.wrench import goal_deadzone_schedule
from ..curriculum_functions.wrench_refactor import arm_velocity_threshold_schedule

@configclass
class WrenchRefactorCurriculumCfg:
    target_goal_direction_deadzone = CurriculumTermCfg(func=goal_deadzone_schedule, params={
        'start_degrees': 30.0, 'start_iteration': 10000, 'interval': 3000, 'decrement_degrees': 5.0,
    })


@configclass
class WrenchRefactorRelativeCurriculumCfg(WrenchRefactorCurriculumCfg):
    """Preserve goal directions and progressively tighten the arm speed cutoff."""
    arm_joint_velocity_threshold = CurriculumTermCfg(func=arm_velocity_threshold_schedule, params={
        'initial': 4.0, 'minimum': 2.0, 'start_iteration': 10000,
        'interval': 250, 'decrement': 0.05,
    })
