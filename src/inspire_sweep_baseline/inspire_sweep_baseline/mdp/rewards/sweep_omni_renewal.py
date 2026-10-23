"""Renewal composition with shared exponential excess-speed penalties.

Renewal mechanics with Dex distance rewards and exponential speed costs.

Dex Tactile v1 with goal occupancy rewards continuing until timeout."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.sweep import _arm
from ..common.sweep_omni import OBJECT_RADIUS
from ..common.sweep_omni import SHELF_HALF_EXTENTS
from ..common.sweep_omni import SHELF_SURFACE_HEIGHT
from ..common.sweep_omni_renewal import FAILURE_TERMS
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_hold_rewards import StableGoalAchievement
from ..reward_functions.goal_rewards import HeightGatedPlanarReachProgress
from ..reward_functions.goal_rewards import PlanarGoalProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_joint_speed_cost
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.goal_rewards import object_tilt_cost
from ..reward_functions.goal_rewards import stable_goal_hold
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math

@configclass
class OmniRenewalRewardsCfg:
    reaching = RewTerm(
        func=HeightGatedPlanarReachProgress,
        weight=30.0,
        params={
            "near_radius": 0.10,
            "frame_name": "ee_frame",
            "z_offset": 0.09,
            "z_tolerance": 0.02,
        },
    )
    goal_distance = RewTerm(
        func=PlanarGoalProgress,
        weight=40.0,
        params={"position_tolerance": 0.01},
    )

    arm_action_rate = RewTerm(
        func=action_rate_l2,
        weight=-0.03,
        params={"action_group": "arm"},
    )
    hand_action_rate = RewTerm(
        func=action_rate_l2,
        weight=-0.03,
        params={"action_group": "hand"},
    )

    arm_joint_velocity = RewTerm(
        func=joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": _arm()},
    )
    hand_joint_velocity = RewTerm(
        func=joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            )
        },
    )

    object_tilt = RewTerm(
        func=object_tilt_cost,
        weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)

    pushing_speed = RewTerm(
        func=exponential_pushing_speed_cost,
        weight=-1.0,
        params={"free_speed": 0.3},
    )
    arm_speed_excess = RewTerm(
        func=exponential_joint_speed_cost,
        weight=-1.0,
        params={"asset_cfg": _arm(), "free_speed": 1.0},
    )


@configclass
class DexOmniRenewalRewardsCfg(OmniRenewalRewardsCfg):
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=300.0,
        params={"position_tolerance": 0.01},
    )


@configclass
class DexTactileOmniRenewalV2RewardsCfg(DexOmniRenewalRewardsCfg):
    # Evaluate the shared timer first; latch +100 without resetting the env.
    success = RewTerm(
        func=StableGoalAchievement,
        weight=100.0,
        params={
            "position_tolerance": 0.01,
            "speed_tolerance": 0.02,
            "upright_angle": math.radians(10.0),
            "hold_time": 0.5,
            "surface_height": SHELF_SURFACE_HEIGHT,
            "support_tolerance": 0.01,
            "half_extents": SHELF_HALF_EXTENTS,
            "object_radius": OBJECT_RADIUS,
            "failure_terms": FAILURE_TERMS,
        },
    )
    goal_hold = RewTerm(func=stable_goal_hold, weight=20.0)
