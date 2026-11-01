"""Independent Revision termination configuration; success occurs only at timeout."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.final import shelf_termination
from ..termination_functions.revision import (
    any_active_object_fall_or_tilt, success_at_timeout, unsuccessful_timeout,
)
from ..termination_functions.sweep import arm_velocity, push_fast


@configclass
class RevisionTerminationsCfg:
    object_fall_or_tilt = DoneTerm(
        func=any_active_object_fall_or_tilt,
        params={"bottom_clearance": 0.01, "fail_angle": 0.9},
    )
    target_speed = DoneTerm(func=push_fast, params={"speed_condition": 1.0})
    shelf_failure = DoneTerm(func=shelf_termination)
    arm_joint_velocity = DoneTerm(func=arm_velocity, params={"asset_cfg": arm_entity(), "threshold": 5.0})
    # Both partitions are truncations, preserving v0_2 value bootstrapping at 10 s.
    target_goal_success_at_timeout = DoneTerm(func=success_at_timeout, time_out=True)
    target_goal_miss_at_timeout = DoneTerm(func=unsuccessful_timeout, time_out=True)
