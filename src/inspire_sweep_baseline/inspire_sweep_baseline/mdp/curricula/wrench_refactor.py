"""Only the inherited 10k/3k direction schedule remains active."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.wrench import goal_deadzone_schedule

@configclass
class WrenchRefactorCurriculumCfg:
    target_goal_direction_deadzone = CurriculumTermCfg(func=goal_deadzone_schedule, params={
        'start_degrees': 30.0, 'start_iteration': 10000, 'interval': 3000, 'decrement_degrees': 5.0,
    })
