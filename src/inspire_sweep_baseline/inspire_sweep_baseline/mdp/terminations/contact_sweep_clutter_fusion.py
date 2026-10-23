"""Explicit Clutter v1 failure conditions and timeout; no success termination.

Explicit v3 failures and timeout, with no success termination.

Retain failures and add success after the existing failure terms."""

from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.contact_sweep_clutter import any_object_fall_or_tilt
from ..termination_functions.contact_sweep_clutter_fusion import GoalHoldSuccess
from ..termination_functions.contact_sweep_clutter_fusion import any_active_object_fall_or_tilt
from ..termination_functions.goal_terminations import shelf_displacement
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.sweep import push_fast
from ..termination_functions.sweep import shelf_collision
from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterFusionTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=any_object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(func=push_fast, params={"speed_condition": 1.0})
    shelf_collision = DoneTerm(
        func=shelf_collision,
        params={"threshold": 0.1, "check_fingers": False, "wrist_clearance": 0.03},
    )
    # Historical metric name retained; the selected joints are the six ARM joints.
    hand_velocity = DoneTerm(
        func=arm_velocity,
        params={"asset_cfg": arm_entity(), "threshold": 5.0},
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=shelf_displacement, params={"threshold": 0.05}
    )


@configclass
class ContactSweepClutterFusionV1_0TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=any_active_object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(
        func=push_fast, params={"speed_condition": 0.8},
        # Monitoring option (disabled with the commented function block): "print_velocity": True
    )
    shelf_collision = DoneTerm(
        func=shelf_collision,
        params={"threshold": 0.1, "check_fingers": False, "wrist_clearance": 0.03},
    )
    # Preserve the existing log key for the six ARM joints.
    hand_velocity = DoneTerm(
        func=arm_velocity,
        params={"asset_cfg": arm_entity(), "threshold": 5.0},
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=shelf_displacement, params={"threshold": 0.05},
    )


@configclass
class ContactSweepClutterFusionV1_4TerminationsCfg(ContactSweepClutterFusionV1_0TerminationsCfg):
    push_fast = ContactSweepClutterFusionV1_0TerminationsCfg().push_fast.replace(
        params={"speed_condition": 1.0},
    )
    success = DoneTerm(
        func=GoalHoldSuccess,
        params={"hold_time": 3.0,
                "failure_terms": ("object_drop", "push_fast", "shelf_collision",
                                  "hand_velocity", "shelf_displacement")},
    )
