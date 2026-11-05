"""Frozen Test-v2 rewards; +75 now pays on settled success."""

import math
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter_fusion import (
    obstacle_contact_force_cost,
    ObjectAccelerationPenalty,
    active_object_tilt_cost,
    active_obstacle_displacement_cost,
)
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.final import (
    ShelfPenalty,
    action_rate_penalty,
    joint_velocity_penalty,
)
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.revision import (
    target_contact_below_height,
    IndependentWrenchMagnitudeChangePenalty,
    goal_distance_v2,
    goal_hold_v2,
    FirstGoalEntryBonusV2,
)

from ..reward_functions.wrench import (
    settled_success_bonus,
    desired_wrench_action_rate_penalty,
    WrenchV0_1GoalDistance,
    goal_distance_cosine,
    WrenchV0_2ContactMaintenance,
)


@configclass
class WrenchRewardsCfg:
    target_reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    target_goal_distance = RewTerm(
        func=goal_distance_v2, weight=3.7, params={"distance_scale": 0.05}
    )
    target_goal_hold = RewTerm(
        func=goal_hold_v2,
        weight=10.0,
        params={"full_reward_radius": 0.025, "outer_radius": 0.035},
    )
    target_contact = RewTerm(
        func=target_contact_below_height,
        weight=1.45,
        params={"threshold_n": 0.01},
    )
    robot_action_rate_penalty = RewTerm(func=action_rate_penalty, weight=-1.0)
    robot_joint_velocity_penalty = RewTerm(
        func=joint_velocity_penalty,
        weight=-1.0,
        params={
            "arm_cfg": arm_entity(),
            "hand_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            ),
        },
    )
    object_tilt_penalty = RewTerm(
        func=active_object_tilt_cost,
        weight=-30.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    obstacle_displacement_penalty = RewTerm(
        func=active_obstacle_displacement_cost,
        weight=-150.0,
        params={"displacement_max_cost": 0.04},
    )
    object_acceleration_penalty = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=-0.05,
        params={"free_acceleration": 15.0, "max_cost": 100.0},
    )
    shelf_penalty = RewTerm(
        func=ShelfPenalty,
        weight=-1.0,
        params={"termination_term_name": "shelf_failure"},
    )
    obstacle_contact_penalty = RewTerm(
        func=obstacle_contact_force_cost,
        weight=-0.05,
        params={"free_force": 50.0, "max_cost": 100.0},
    )
    target_goal_first_entry_bonus = RewTerm(
        func=FirstGoalEntryBonusV2,
        weight=25.0,
        params={"free_speed": 0.01, "speed_scale": 0.02},
    )
    target_goal_success_bonus = RewTerm(func=settled_success_bonus, weight=75.0)
    wrist_wrench_change_penalty = RewTerm(
        func=IndependentWrenchMagnitudeChangePenalty,
        weight=-0.1,
        params={
            "free_force_change": 20.0,
            "free_moment_change": 2.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
        },
    )
    target_speed_penalty = RewTerm(
        func=exponential_pushing_speed_cost_limit,
        weight=-5.0,
        params={"free_speed": 0.1, "max_cost": 100.0},
    )


@configclass
class WrenchV0_1RewardsCfg(WrenchRewardsCfg):
    target_contact = None
    target_goal_distance = RewTerm(
        func=WrenchV0_1GoalDistance,
        weight=3.7,
        params={"distance_scale": 0.05, "threshold_n": 0.01},
    )


@configclass
class WrenchV1RewardsCfg(WrenchRewardsCfg):
    robot_action_rate_penalty = RewTerm(
        func=desired_wrench_action_rate_penalty, weight=-1.0
    )


@configclass
class WrenchV0_2RewardsCfg(WrenchRewardsCfg):
    target_goal_distance = RewTerm(
        func=goal_distance_cosine, weight=3.7, params={"distance_scale": 0.05}
    )
    target_contact = RewTerm(
        func=WrenchV0_2ContactMaintenance, weight=0.1, params={"threshold_n": 0.01}
    )
