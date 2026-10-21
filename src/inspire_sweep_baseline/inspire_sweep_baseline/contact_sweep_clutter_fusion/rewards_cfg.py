"""Explicit fused rewards; weights apply once through RewardManager and dt."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..assets.robot import HAND_JOINT_NAMES
from ..mdp import dex_rewards, goal_rewards
from ..mdp.rewards import ShelfCollisionPenalty
from ..contact_sweep_clutter_v1.mdp import (
    contact_any,
    contact_count,
    goal_hold_xy,
    shelf_failure,
)
from .common import arm_entity
from .reward_functions import (
    combined_action_rate,
    combined_joint_velocity,
    combined_object_cost,
)
from .safety_rewards import (
    WrenchRatePenalty,
    eef_linear_speed_cost,
    eef_upward_z_orientation_cost,
)


@configclass
class ddContactSweepClutterFusionRewardsCfg:
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
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    contact_any = RewTerm(func=contact_any, weight=0.4)
    contact_count = RewTerm(func=contact_count, weight=0.4, params={"max_contacts": 5})
    # The merged action and velocity costs retain the original per-group caps.
    action_rate = RewTerm(
        func=combined_action_rate, weight=-0.03, params={"max_cost": 100.0}
    )
    joint_velocity = RewTerm(
        func=combined_joint_velocity,
        weight=-0.03,
        params={
            "arm_cfg": arm_entity(),
            "hand_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            ),
            "hand_ratio": 1.0 / 3.0,
            "max_cost": 100.0,
        },
    )
    # -30*(tilt + 2.5*displacement) == 15*(-2*tilt -5*displacement).
    object = RewTerm(
        func=combined_object_cost,
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
    # At most -0.2 per 10-second episode, ~1.17% of the logged mean return 17.13.
    eef_orientation = RewTerm(
        func=eef_upward_z_orientation_cost,
        weight=-0.02,
        params={"onset_angle": math.pi / 3.0},
    )
    # Replaces v2's quadratic change cost. Must follow all positive rewards.
    wrench_change = RewTerm(
        func=WrenchRatePenalty,
        weight=-0.1,
        params={
            "free_force_rate": 2000.0,
            "free_moment_rate": 100.0,
            "saturation_force_rate": 5000.0,
            "saturation_moment_rate": 250.0,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


# Keep the existing working-tree class name usable through the public import.
ContactSweepClutterFusionRewardsCfg = ddContactSweepClutterFusionRewardsCfg
