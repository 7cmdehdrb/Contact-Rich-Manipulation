"""Self-contained baseline configuration, derived from the v2-mass behavior."""

from ..common.sweep import TARGET_Z_OFFSET
from ..common.sweep import _arm
from ..reward_functions.sweep import ShelfCollisionPenalty
from ..reward_functions.sweep import action_rate_l2
from ..reward_functions.sweep import align_eef_y_with_shelf_z
from ..reward_functions.sweep import homing_reward
from ..reward_functions.sweep import joint_vel_l2
from ..reward_functions.sweep import object_collision
from ..reward_functions.sweep import pushing_target
from ..reward_functions.sweep import reaching_target
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

@configclass
class RewardsCfg:
    action_rate = RewTerm(func=action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(func=joint_vel_l2, weight=-0.03, params={"asset_cfg": _arm()})
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    object_collision = RewTerm(func=object_collision, weight=-0.5)
    reaching = RewTerm(func=reaching_target, weight=3.0, params={"z_offset": TARGET_Z_OFFSET})
    orientation = RewTerm(func=align_eef_y_with_shelf_z, weight=2.0)
    sweeping_object = RewTerm(
        func=pushing_target, weight=6.0,
        params={"command_name": "target_goal_pos", "z_offset": TARGET_Z_OFFSET},
    )
    homing_after_sweep = RewTerm(
        func=homing_reward, weight=9.0,
        params={"command_name": "target_goal_pos", "asset_cfg": _arm()},
    )
