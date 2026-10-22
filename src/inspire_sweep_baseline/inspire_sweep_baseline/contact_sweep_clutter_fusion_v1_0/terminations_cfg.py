"""Explicit v3 failures and timeout, with no success termination."""

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.common import arm_entity
from ..mdp import goal_terminations, terminations
from .object_state import any_active_object_fall_or_tilt


@configclass
class ContactSweepClutterFusionV1_0TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=any_active_object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(func=terminations.push_fast, params={"speed_condition": 0.8})
    shelf_collision = DoneTerm(
        func=terminations.shelf_collision,
        params={"threshold": 0.1, "check_fingers": False, "wrist_clearance": 0.03},
    )
    # Preserve the existing log key for the six ARM joints.
    hand_velocity = DoneTerm(
        func=terminations.arm_velocity,
        params={"asset_cfg": arm_entity(), "threshold": 5.0},
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=goal_terminations.shelf_displacement, params={"threshold": 0.05},
    )
