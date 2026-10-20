"""All ten active rewards are declared here, without inherited overrides."""

import math
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..assets.robot import HAND_JOINT_NAMES
from ..mdp import dex_rewards, goal_rewards
from ..mdp.rewards import ShelfCollisionPenalty
from .common import arm_entity


@configclass
class ContactSweepRewardsCfg:
    reaching = RewTerm(
        func=dex_rewards.DexReachProgress, weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    # v1 run 2026-10-19_12-15-47, steps 39027–39036: empirical 3:4 balance.
    goal_distance = RewTerm(
        func=dex_rewards.DexPlanarGoalProgress, weight=94.0,
        params={"position_tolerance": 0.01},
    )
    arm_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2, weight=-0.03, params={"action_group": "arm"},
    )
    hand_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2, weight=-0.03, params={"action_group": "hand"},
    )
    arm_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2, weight=-0.03, params={"asset_cfg": arm_entity()},
    )
    hand_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
    )
    object_tilt = RewTerm(
        func=goal_rewards.object_tilt_cost, weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(
        func=goal_rewards.exponential_pushing_speed_cost_limit, weight=-2.0,
        params={"free_speed": 0.2, "max_cost": 100.0},
    )
    arm_speed_excess = RewTerm(
        func=goal_rewards.exponential_joint_speed_cost_limit, weight=-1.0,
        params={"asset_cfg": arm_entity(), "free_speed": 1.0, "max_cost": 100.0},
    )
