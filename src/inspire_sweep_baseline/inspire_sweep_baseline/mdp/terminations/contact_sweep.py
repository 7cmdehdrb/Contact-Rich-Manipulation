"""Boolean failure and timeout terms; success does not terminate episodes."""

from ..common.contact_sweep import arm_entity
from ..termination_functions.goal_terminations import object_fall_or_tilt
from ..termination_functions.goal_terminations import shelf_displacement
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.sweep import push_fast
from ..termination_functions.sweep import shelf_collision
from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

@configclass
class ContactSweepTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(
        func=object_fall_or_tilt, params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(func=push_fast, params={"speed_condition": 0.8})
    shelf_collision = DoneTerm(
        func=shelf_collision,
        params={"threshold": 0.1, "check_fingers": False, "wrist_clearance": 0.03},
    )
    # Keep the historical metric name: this term checks the six ARM joints.
    hand_velocity = DoneTerm(
        func=arm_velocity, params={"asset_cfg": arm_entity(), "threshold": 5.0},
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(func=shelf_displacement, params={"threshold": 0.05})
