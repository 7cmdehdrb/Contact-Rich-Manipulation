"""All ten active rewards are declared here, without inherited overrides."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep import arm_entity
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_joint_speed_cost_limit
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.goal_rewards import object_tilt_cost
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math

@configclass
class ContactSweepRewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress, weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    # v1 run 2026-10-19_12-15-47, steps 39027–39036: empirical 3:4 balance.
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress, weight=94.0,
        params={"position_tolerance": 0.01},
    )
    arm_action_rate = RewTerm(
        func=action_rate_l2, weight=-0.03, params={"action_group": "arm"},
    )
    hand_action_rate = RewTerm(
        func=action_rate_l2, weight=-0.03, params={"action_group": "hand"},
    )
    arm_joint_velocity = RewTerm(
        func=joint_vel_l2, weight=-0.03, params={"asset_cfg": arm_entity()},
    )
    hand_joint_velocity = RewTerm(
        func=joint_vel_l2, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
    )
    object_tilt = RewTerm(
        func=object_tilt_cost, weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(
        func=exponential_pushing_speed_cost_limit, weight=-2.0,
        params={"free_speed": 0.2, "max_cost": 100.0},
    )
    arm_speed_excess = RewTerm(
        func=exponential_joint_speed_cost_limit, weight=-1.0,
        params={"asset_cfg": arm_entity(), "free_speed": 1.0, "max_cost": 100.0},
    )
