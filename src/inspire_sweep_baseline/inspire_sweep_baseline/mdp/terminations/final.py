"""Final v0 failure and timeout terms; no success termination."""

from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.contact_sweep_clutter_fusion import any_active_object_fall_or_tilt
from ..termination_functions.goal_terminations import shelf_displacement
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.sweep import push_fast
from ..termination_functions.sweep import shelf_collision
from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


@configclass
class FinalTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=any_active_object_fall_or_tilt,
        params={'height_condition': 1.04, 'fail_angle': 0.9},
    )
    push_fast = DoneTerm(func=push_fast, params={'speed_condition': 1.0})
    shelf_collision = DoneTerm(
        func=shelf_collision,
        params={'threshold': 0.1, 'check_fingers': False, 'wrist_clearance': 0.03},
    )
    hand_velocity = DoneTerm(func=arm_velocity, params={'asset_cfg': arm_entity(), 'threshold': 5.0})
    object_outside_shelf = None
    shelf_displacement = DoneTerm(func=shelf_displacement, params={'threshold': 0.05})
    success = None


from ..termination_functions.final import shelf_termination, success_at_timeout, unsuccessful_timeout
from ..termination_functions.final import goal_accuracy_success_at_timeout, goal_accuracy_unsuccessful_timeout


@configclass
class FinalV1TerminationsCfg:
    object_drop = FinalTerminationsCfg().object_drop
    object_speed = FinalTerminationsCfg().push_fast
    shelf_termination = DoneTerm(func=shelf_termination)
    joint_velocity = FinalTerminationsCfg().hand_velocity
    # Both partitions remain truncations: preserve value bootstrapping at 10 s.
    success = DoneTerm(func=success_at_timeout, time_out=True)
    time_out = DoneTerm(func=unsuccessful_timeout, time_out=True)


@configclass
class FinalGoalAccuracyTerminationsCfg(FinalV1TerminationsCfg):
    success = DoneTerm(func=goal_accuracy_success_at_timeout, time_out=True)
    time_out = DoneTerm(func=goal_accuracy_unsuccessful_timeout, time_out=True)


@configclass
class FinalGoalAccuracyV0_1TerminationsCfg(FinalGoalAccuracyTerminationsCfg):
    object_drop = None
    object_speed = None
    shelf_termination = None
    joint_velocity = None
    success = None
    time_out = None
    object_fall_or_tilt = FinalGoalAccuracyTerminationsCfg().object_drop
    target_speed = FinalGoalAccuracyTerminationsCfg().object_speed
    shelf_failure = FinalGoalAccuracyTerminationsCfg().shelf_termination
    arm_joint_velocity = FinalGoalAccuracyTerminationsCfg().joint_velocity
    target_goal_success_at_timeout = FinalGoalAccuracyTerminationsCfg().success.replace(
        params={'goal_hold_term_name': 'target_goal_hold'})
    target_goal_miss_at_timeout = FinalGoalAccuracyTerminationsCfg().time_out.replace(
        params={'goal_hold_term_name': 'target_goal_hold'})
