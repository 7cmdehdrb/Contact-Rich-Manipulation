"""Explicit Final v0 terms, frozen from Holding OSC ObjectRandom v3."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter import shelf_failure
from ..reward_functions.contact_sweep_clutter_fusion import ObjectAccelerationPenalty
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import active_object_tilt_cost
from ..reward_functions.contact_sweep_clutter_fusion import (
    active_obstacle_displacement_cost,
)
from ..reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
from ..reward_functions.contact_sweep_clutter_fusion import eef_linear_speed_cost
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_joint_speed_cost_limit
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.goal_rewards import success_event
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import (
    hand_contact_or_goal,
)
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import (
    goal_settling_centered,
)
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import (
    ReachingOrGoal,
)


@configclass
class FinalRewardsCfg:
    reaching = RewTerm(
        func=ReachingOrGoal,
        weight=30.0,
        params={"near_radius": 0.1, "frame_name": "ee_frame", "goal_near_radius": 1.0},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress, weight=100.0, params={"position_tolerance": 0.0}
    )
    goal_hold = RewTerm(
        func=goal_settling_centered,
        weight=4.0,
        params={"position_tolerance": 0.02, "speed_scale": 0.05},
    )
    hand_contact = RewTerm(
        func=hand_contact_or_goal, weight=1.45, params={"threshold_n": 0.01}
    )
    arm_action_rate = RewTerm(
        func=action_rate_l2,
        weight=-0.03,
        params={"action_group": "arm", "max_cost": 100.0},
    )
    hand_action_rate = RewTerm(
        func=action_rate_l2,
        weight=-0.03,
        params={"action_group": "hand", "max_cost": 100.0},
    )
    arm_joint_velocity = RewTerm(
        func=joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": arm_entity(), "max_cost": 100.0},
    )
    hand_joint_velocity = RewTerm(
        func=joint_vel_l2,
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
            "displacement_ratio": 5.0,
            "displacement_max_cost": 0.04,
        },
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(
        func=exponential_pushing_speed_cost_limit,
        weight=-5.0,
        params={"free_speed": 0.1, "max_cost": 100.0},
    )
    arm_speed_excess = RewTerm(
        func=exponential_joint_speed_cost_limit,
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
        weight=0.0,
        params={"free_force": 80.0, "max_cost": 100.0},
    )
    object_acceleration = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=0.0,
        params={"free_acceleration": 50.0, "max_cost": 100.0},
    )
    wrench_change = RewTerm(
        func=WrenchMagnitudeChangePenalty,
        weight=0.0,
        params={
            "free_force_change": 50.0,
            "free_moment_change": 5.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


from ..reward_functions.final import (
    action_rate_penalty,
    joint_velocity_penalty,
    ObjectPenalty,
    ShelfPenalty,
)
from ..reward_functions.final import exponential_planar_goal_proximity
from ..reward_functions.final import goal_accuracy_settling_centered


@configclass
class FinalV1RewardsCfg:
    """Negative outer weights keep wrench's positive-reward budget unchanged."""

    hand_reaching = FinalRewardsCfg().reaching
    goal_distance = FinalRewardsCfg().goal_distance
    goal_hold = FinalRewardsCfg().goal_hold
    hand_contact = FinalRewardsCfg().hand_contact
    action_rate_penalty = RewTerm(func=action_rate_penalty, weight=-1.0)
    joint_velocity_penalty = RewTerm(
        func=joint_velocity_penalty,
        weight=-1.0,
        params={
            "arm_cfg": arm_entity(),
            "hand_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            ),
        },
    )
    object_penalty = RewTerm(
        func=ObjectPenalty,
        weight=-1.0,
        params={
            **FinalRewardsCfg().object.params,
            "free_acceleration": 50.0,
            "acceleration_weight": 0.0,
            "max_cost": 100.0,
        },
    )
    object_speed_penalty = FinalRewardsCfg().pushing_speed
    shelf_penalty = RewTerm(func=ShelfPenalty, weight=-1.0)
    obstacle_contact_penalty = FinalRewardsCfg().obstacle_contact
    wrench_change_penalty = FinalRewardsCfg().wrench_change


@configclass
class FinalV1_1RewardsCfg(FinalV1RewardsCfg):
    """Keep v1 terms and pay exponential XY proximity at every policy step."""

    goal_distance = RewTerm(
        func=exponential_planar_goal_proximity,
        weight=1.5,
        params={"distance_scale": 0.05},
    )


@configclass
class FinalV1TestRewardsCfg(FinalV1RewardsCfg):
    """Expose the three v1 object costs separately without changing their sum."""

    object_penalty = None
    object_tilt_penalty = RewTerm(
        func=active_object_tilt_cost,
        weight=-30.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    object_displacement_penalty = RewTerm(
        func=active_obstacle_displacement_cost,
        weight=-150.0,
        params={"displacement_max_cost": 0.04},
    )
    object_acceleration_penalty = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=0.0,
        params={"free_acceleration": 50.0, "max_cost": 100.0},
    )


@configclass
class FinalGoalAccuracyRewardsCfg(FinalV1RewardsCfg):
    """Keep v1 reaching and speed costs with a fixed, precise hold radius."""

    goal_hold = RewTerm(
        func=goal_accuracy_settling_centered,
        weight=8.0,
        params={"position_tolerance": 0.01, "speed_scale": 0.05},
    )
    object_speed_penalty = None
    target_speed_penalty = FinalV1RewardsCfg().object_speed_penalty


@configclass
class FinalGoalAccuracyV1RewardsCfg(FinalGoalAccuracyRewardsCfg):
    """Restore Final v1's outer hold reward while retaining v0's radius and weight."""

    goal_hold = FinalGoalAccuracyRewardsCfg().goal_hold.replace(
        func=goal_settling_centered
    )


from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.final import (
    goal_accuracy_outer_settling,
    target_contact_or_goal,
    target_contact_only,
    target_goal_timeout_bonus,
)


@configclass
class FinalGoalAccuracyV0_1RewardsCfg(FinalGoalAccuracyRewardsCfg):
    """Two-centimeter target control; renamed terms retain parent execution order."""

    hand_reaching = None
    goal_distance = None
    goal_hold = None
    hand_contact = None
    action_rate_penalty = None
    joint_velocity_penalty = None
    wrench_change_penalty = None
    target_reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    target_goal_distance = RewTerm(
        func=exponential_planar_goal_proximity,
        weight=3.7,
        params={"distance_scale": 0.05},
    )
    target_goal_hold = RewTerm(
        func=goal_accuracy_outer_settling,
        weight=8.0,
        params={"position_tolerance": 0.02, "outer_radius": 0.06, "speed_scale": 0.05},
    )
    target_contact = FinalGoalAccuracyRewardsCfg().hand_contact.replace(
        func=target_contact_or_goal
    )
    robot_action_rate_penalty = FinalGoalAccuracyRewardsCfg().action_rate_penalty
    robot_joint_velocity_penalty = FinalGoalAccuracyRewardsCfg().joint_velocity_penalty
    wrist_wrench_change_penalty = FinalGoalAccuracyRewardsCfg().wrench_change_penalty
    shelf_penalty = FinalGoalAccuracyRewardsCfg().shelf_penalty.replace(
        params={"termination_term_name": "shelf_failure"}
    )

    def __post_init__(self):
        # Dataclasses put renamed child fields after inherited fields. Managers
        # iterate __dict__, so explicitly restore the parent's active order.
        order = (
            "target_reaching",
            "target_goal_distance",
            "target_goal_hold",
            "target_contact",
            "robot_action_rate_penalty",
            "robot_joint_velocity_penalty",
            "object_penalty",
            "shelf_penalty",
            "obstacle_contact_penalty",
            "wrist_wrench_change_penalty",
            "target_speed_penalty",
        )
        values = {name: self.__dict__[name] for name in order}
        values.update(self.__dict__)
        self.__dict__.clear()
        self.__dict__.update(values)


@configclass
class FinalGoalAccuracyV0_2RewardsCfg(FinalGoalAccuracyV0_1RewardsCfg):
    """Preserve v0_1 and add a one-shot +30 reward at a successful timeout."""

    target_goal_timeout_bonus = RewTerm(
        func=target_goal_timeout_bonus,
        weight=30.0,
        params={"position_tolerance": 0.02},
    )

    def __post_init__(self):
        super().__post_init__()
        # Wrench's cumulative budget must see every positive reward this step.
        bonus = self.__dict__.pop("target_goal_timeout_bonus")
        values = {}
        for name, value in self.__dict__.items():
            if name == "wrist_wrench_change_penalty":
                values["target_goal_timeout_bonus"] = bonus
            values[name] = value
        self.__dict__.clear()
        self.__dict__.update(values)


@configclass
class FinalGoalAccuracyV0_3RewardsCfg(FinalGoalAccuracyV0_2RewardsCfg):
    """Three-centimeter success with center-seeking progress and a +100 timeout."""

    target_goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.0},
    )
    target_goal_hold = FinalGoalAccuracyV0_2RewardsCfg().target_goal_hold.replace(
        params={"position_tolerance": 0.03, "outer_radius": 0.06, "speed_scale": 0.05}
    )
    target_contact = FinalGoalAccuracyV0_2RewardsCfg().target_contact.replace(weight=0.3625)
    target_goal_timeout_bonus = RewTerm(
        func=success_event,
        weight=100.0,
        params={"termination_name": "target_goal_success_at_timeout"},
    )


@configclass
class FinalGoalAccuracyV0_2_1RewardsCfg(FinalGoalAccuracyV0_2RewardsCfg):
    """Actual target contact and an inclusive, one-shot +75 timeout success."""

    target_contact = FinalGoalAccuracyV0_2RewardsCfg().target_contact.replace(func=target_contact_only)
    target_goal_timeout_bonus = RewTerm(
        func=success_event, weight=75.0,
        params={"termination_name": "target_goal_success_at_timeout"},
    )
