"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards.

OmniV2 mechanics with DexTouch-style distance-record reward shaping."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.sweep import _arm
from ..common.sweep_omni import FAILURE_TERMS
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_rewards import CurrentObjectReachProgress
from ..reward_functions.goal_rewards import PlanarGoalProgress
from ..reward_functions.goal_rewards import ReachProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import failure_event
from ..reward_functions.goal_rewards import joint_speed_cost
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.goal_rewards import object_tilt_cost
from ..reward_functions.goal_rewards import planar_goal_cost
from ..reward_functions.goal_rewards import pushing_speed_cost
from ..reward_functions.goal_rewards import success_event
from ..reward_functions.goal_rewards import time_cost
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math

@configclass
class OmniRewardsCfg:
    # Dense costs are per second: RewardManager applies step_dt (T_ref = 1 s).

    # 물체 근처로 이동하는 Reaching 보상
    reaching = RewTerm(
        func=ReachProgress,
        weight=1.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    # 물체를 목표 위치로 이동시키는 Goal Distance 보상
    goal_distance = RewTerm(
        func=planar_goal_cost, weight=6.0, params={"sigma": 0.10}
    )

    # Mandatory Inspire-compatible squared sums; keep Arm/Hand costs separate.
    # Raw action differences (not action magnitudes or clipped Hand commands).
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
            ),
        },
    )

    # 물체가 떨어지거나 기울어지는 것을 방지하는 보상
    object_tilt = RewTerm(
        func=object_tilt_cost,
        weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    # 물체가 선반 밖으로 나가는 것을 방지하는 보상
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)

    # 물체를 목표 위치로 이동시키는 것을 성공으로 간주하는 보상
    time = RewTerm(func=time_cost, weight=-0.1)

    # Progress and terminal functions undo the manager's dt multiplication.
    success = RewTerm(func=success_event, weight=30.0)
    failure = RewTerm(
        func=failure_event,
        weight=-80.0,
        params={"failure_terms": FAILURE_TERMS},
    )


@configclass
class OmniV2RewardsCfg(OmniRewardsCfg):
    # Experimental budgets per episode, independent of dt and episode length.
    # Before contact, reaching provides the main positive learning signal.
    # After approach is exhausted, only object-goal progress pays task reward.
    reaching = RewTerm(
        func=CurrentObjectReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=PlanarGoalProgress,
        weight=40.0,
        params={"position_tolerance": 0.01},
    )
    time = None
    success = None
    failure = None


@configclass
class DexOmniRewardsCfg(OmniV2RewardsCfg):
    # Experimental sweep coefficients in reward/meter, not paper parameters or
    # normalized episode budgets. A 14 cm goal improvement pays 42 reward.
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
class OmniV4RewardsCfg(OmniV2RewardsCfg):
    pushing_speed = RewTerm(
        func=pushing_speed_cost,
        weight=-0.05,
        params={"free_speed": 0.3, "reference_speed": 0.5},
    )
    arm_speed_excess = RewTerm(
        func=joint_speed_cost,
        weight=-0.03,
        params={"asset_cfg": _arm(), "free_speed": 1.0, "reference_speed": 2.0},
    )
