"""Explicit Revision rewards, preserving v0_2 costs and reward execution order."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..common.revision_goal import GOAL_SUCCESS_RADIUS, HOLD_OUTER_RADIUS
from ..reward_functions.contact_sweep_clutter_fusion import (
    WrenchMagnitudeChangePenalty, obstacle_contact_force_cost, ObjectAccelerationPenalty,
    active_object_tilt_cost, active_obstacle_displacement_cost,
)
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.final import (
    ObjectPenalty, ShelfPenalty, action_rate_penalty, exponential_planar_goal_proximity,
    joint_velocity_penalty,
)
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.revision import (
    goal_hold, successful_timeout_bonus, target_contact_below_height, goal_hold_v1,
    FirstGoalEntryBonus, IndependentWrenchMagnitudeChangePenalty,
)


@configclass
class RevisionRewardsCfg:
    target_reaching = RewTerm(
        func=DexReachProgress, weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    target_goal_distance = RewTerm(
        func=exponential_planar_goal_proximity, weight=3.7,
        params={"distance_scale": 0.05},
    )
    target_goal_hold = RewTerm(
        func=goal_hold, weight=8.0,
        params={"position_tolerance": GOAL_SUCCESS_RADIUS, "outer_radius": HOLD_OUTER_RADIUS, "speed_scale": 0.05},
    )
    target_contact = RewTerm(
        func=target_contact_below_height, weight=1.45, params={"threshold_n": 0.01},
    )
    robot_action_rate_penalty = RewTerm(func=action_rate_penalty, weight=-1.0)
    robot_joint_velocity_penalty = RewTerm(
        func=joint_velocity_penalty, weight=-1.0,
        params={
            "arm_cfg": arm_entity(),
            "hand_cfg": SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True),
        },
    )
    object_penalty = RewTerm(
        func=ObjectPenalty, weight=-1.0,
        params={
            "free_angle": math.radians(5.0), "fail_angle": 0.9,
            "displacement_ratio": 5.0, "displacement_max_cost": 0.04,
            "free_acceleration": 50.0, "acceleration_weight": 0.0, "max_cost": 100.0,
        },
    )
    shelf_penalty = RewTerm(
        func=ShelfPenalty, weight=-1.0, params={"termination_term_name": "shelf_failure"},
    )
    obstacle_contact_penalty = RewTerm(
        func=obstacle_contact_force_cost, weight=0.0,
        params={"free_force": 80.0, "max_cost": 100.0},
    )
    # The wrench budget reads every positive reward accumulated this step.
    target_goal_timeout_bonus = RewTerm(func=successful_timeout_bonus, weight=30.0)
    wrist_wrench_change_penalty = RewTerm(
        func=WrenchMagnitudeChangePenalty, weight=0.0,
        params={
            "free_force_change": 50.0, "free_moment_change": 5.0,
            "saturation_ratio": 2.5, "sharpness": 3.0, "max_positive_fraction": 0.05,
        },
    )
    target_speed_penalty = RewTerm(
        func=exponential_pushing_speed_cost_limit, weight=-5.0,
        params={"free_speed": 0.1, "max_cost": 100.0},
    )


@configclass
class RevisionV1RewardsCfg(RevisionRewardsCfg):
    object_penalty = None
    object_tilt_penalty = RewTerm(
        func=active_object_tilt_cost, weight=-30.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    obstacle_displacement_penalty = RewTerm(
        func=active_obstacle_displacement_cost, weight=-150.0,
        params={"displacement_max_cost": 0.04},
    )
    object_acceleration_penalty = RewTerm(
        func=ObjectAccelerationPenalty, weight=-0.05,
        params={"free_acceleration": 20.0, "max_cost": 100.0},
    )
    target_goal_hold = RewTerm(
        func=goal_hold_v1, weight=8.0,
        params={"position_tolerance": GOAL_SUCCESS_RADIUS, "free_speed": 0.05, "speed_scale": 0.05},
    )
    obstacle_contact_penalty = RewTerm(
        func=obstacle_contact_force_cost, weight=-0.05,
        params={"free_force": 50.0, "max_cost": 100.0},
    )
    target_goal_first_entry_bonus = RewTerm(func=FirstGoalEntryBonus, weight=30.0)
    target_goal_timeout_bonus = RewTerm(func=successful_timeout_bonus, weight=40.0)
    wrist_wrench_change_penalty = RewTerm(
        func=IndependentWrenchMagnitudeChangePenalty, weight=-0.1,
        params={"free_force_change": 20.0, "free_moment_change": 2.0,
                "saturation_ratio": 2.5, "sharpness": 3.0},
    )

    def __post_init__(self):
        # RewardManager follows instance insertion order. Replace the combined
        # slot with its three components and insert entry just before timeout.
        terms = dict(self.__dict__)
        self.__dict__.clear()
        added = {"object_tilt_penalty", "obstacle_displacement_penalty",
                 "object_acceleration_penalty", "target_goal_first_entry_bonus"}
        for name, term in terms.items():
            if name in added:
                continue
            if name == "object_penalty":
                for component in ("object_tilt_penalty", "obstacle_displacement_penalty",
                                  "object_acceleration_penalty"):
                    self.__dict__[component] = terms[component]
            if name == "target_goal_timeout_bonus":
                self.__dict__["target_goal_first_entry_bonus"] = terms["target_goal_first_entry_bonus"]
            self.__dict__[name] = term
