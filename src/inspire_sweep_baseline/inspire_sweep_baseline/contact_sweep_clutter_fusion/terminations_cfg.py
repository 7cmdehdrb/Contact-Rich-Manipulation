"""Explicit Clutter v1 failure conditions and timeout; no success termination."""

from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..contact_sweep_clutter.mdp import any_object_fall_or_tilt
from ..mdp import goal_terminations, terminations
from .common import arm_entity


@configclass
class ContactSweepClutterFusionTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=any_object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(func=terminations.push_fast, params={"speed_condition": 1.0})
    shelf_collision = DoneTerm(
        func=terminations.shelf_collision,
        params={"threshold": 0.1, "check_fingers": False, "wrist_clearance": 0.03},
    )
    # Historical metric name retained; the selected joints are the six ARM joints.
    hand_velocity = DoneTerm(
        func=terminations.arm_velocity,
        params={"asset_cfg": arm_entity(), "threshold": 5.0},
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=goal_terminations.shelf_displacement, params={"threshold": 0.05}
    )
