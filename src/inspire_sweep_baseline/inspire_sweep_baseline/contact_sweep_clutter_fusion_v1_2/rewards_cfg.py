"""Explicit reward contract for the v1_2 branch; weights are rates per second."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..assets.robot import HAND_JOINT_NAMES
from ..contact_sweep_clutter_fusion.common import arm_entity
from ..contact_sweep_clutter_fusion.safety_rewards import eef_linear_speed_cost
from ..contact_sweep_clutter_v1.mdp import goal_hold_xy, shelf_failure
from ..mdp import dex_rewards, goal_rewards
from ..mdp.rewards import ShelfCollisionPenalty
from ..contact_sweep_clutter_fusion_v1_0.object_state import combined_active_object_cost
from ..contact_sweep_clutter_fusion_v1_0.reward_functions import (
    ObjectAccelerationPenalty,
    WrenchMagnitudeChangePenalty,
    hand_contact_any,
    obstacle_contact_force_cost,
)


@configclass
class ContactSweepClutterFusionV1_2RewardsCfg:
    reaching = RewTerm(
        func=dex_rewards.DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=dex_rewards.DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.02}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.

    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.01},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
    arm_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2,
        weight=-0.03,
        params={"action_group": "arm", "max_cost": 100.0},
    )
    hand_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2,
        weight=-0.03,
        params={"action_group": "hand", "max_cost": 100.0},
    )
    arm_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": arm_entity(), "max_cost": 100.0},
    )
    hand_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            ),
            "max_cost": 100.0,
        },
    )
    object = RewTerm(
        func=combined_active_object_cost,
        weight=-30.0,
        params={
            "free_angle": math.radians(5.0),
            "fail_angle": 0.9,
            "displacement_ratio": 2.5,
            "displacement_max_cost": 0.04,
        },
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(
        func=goal_rewards.exponential_pushing_speed_cost_limit,
        weight=-5.0,
        params={"free_speed": 0.1, "max_cost": 100.0},
    )
    arm_speed_excess = RewTerm(
        func=goal_rewards.exponential_joint_speed_cost_limit,
        weight=-1.0,
        params={"asset_cfg": arm_entity(), "free_speed": 1.0, "max_cost": 100.0},
    )
    shelf_failure = RewTerm(func=shelf_failure, weight=-3.0)
    eef_linear_speed = RewTerm(
        func=eef_linear_speed_cost,
        weight=-1.0,
        params={"free_speed": 1.5, "max_cost": 100.0},
    )
    eef_orientation = None

    obstacle_contact = RewTerm(
        func=obstacle_contact_force_cost,
        weight=-1.0,
        params={"free_force": 100.0, "max_cost": 100.0},
    )
    object_acceleration = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=-1.0,
        params={"free_acceleration": 100.0, "max_cost": 100.0},
    )

    # Preserve the v3 normalized exponential, absolute cap and positive budget.
    wrench_change = RewTerm(
        func=WrenchMagnitudeChangePenalty,
        weight=-0.1,
        params={
            "free_force_change": 500.0,
            "free_moment_change": 50.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )
