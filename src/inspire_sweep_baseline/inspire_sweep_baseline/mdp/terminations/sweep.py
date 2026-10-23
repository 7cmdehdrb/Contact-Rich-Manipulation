"""Self-contained baseline configuration, derived from the v2-mass behavior."""

from ..common.sweep import _arm
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.sweep import object_drop
from ..termination_functions.sweep import push_fast
from ..termination_functions.sweep import shelf_collision
from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    object_drop = DoneTerm(func=object_drop, params={"height_condition": 1.04, "rotation_condition": 0.9})
    push_fast = DoneTerm(func=push_fast, params={"speed_condition": 0.3})
    shelf_collision = DoneTerm(func=shelf_collision, params={"threshold": 0.1})
    # Historical term name retained in logs; it restricts the six ARM joints.
    hand_velocity = DoneTerm(func=arm_velocity, params={"asset_cfg": _arm(), "threshold": 1.0})
