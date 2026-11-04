"""Wrench failures, settled terminal success, and non-success time limit."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.final import shelf_termination
from ..termination_functions.revision import any_active_object_fall_or_tilt
from ..termination_functions.sweep import arm_velocity, push_fast
from ..termination_functions.wrench import SettledGoalSuccess, unsuccessful_timeout


@configclass
class WrenchTerminationsCfg:
    object_fall_or_tilt = DoneTerm(func=any_active_object_fall_or_tilt,
                                 params={'bottom_clearance': 0.01, 'fail_angle': 0.9})
    target_speed = DoneTerm(func=push_fast, params={'speed_condition': 1.0})
    shelf_failure = DoneTerm(func=shelf_termination)
    arm_joint_velocity = DoneTerm(func=arm_velocity, params={'asset_cfg': arm_entity(), 'threshold': 5.0})
    target_goal_success = DoneTerm(func=SettledGoalSuccess,
                                  params={'radius': 0.03, 'speed_limit': 0.005, 'duration': 3.0})
    target_goal_miss_at_timeout = DoneTerm(func=unsuccessful_timeout, time_out=True)
