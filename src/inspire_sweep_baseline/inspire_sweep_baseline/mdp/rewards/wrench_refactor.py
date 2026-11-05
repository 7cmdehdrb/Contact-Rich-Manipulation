"""Independent Refactor reward table; reuse verified Wrench-family callables."""

import math
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter_fusion import (
    obstacle_contact_force_cost,
    active_object_tilt_cost,
    active_obstacle_displacement_cost,
)
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.final import (
    ShelfPenalty,
    joint_velocity_penalty,
)
from ..reward_functions.revision import (
    IndependentWrenchMagnitudeChangePenalty,
    goal_distance_v2,
    goal_hold_v2,
)

from ..reward_functions.wrench import settled_success_bonus
from ..reward_functions.wrench_refactor import RefactorContactMaintenance, RefactorActionRatePenalty, RefactorFirstEntry


# Independent joints in the vendor right-hand URDF. USD keeps legacy left names.
# Thumb 3/4 and the four distal finger joints are mimic followers.
HAND_DRIVE_JOINT_NAMES = (
    "inspire_left_thumb_1_joint", "inspire_left_thumb_2_joint",
    "inspire_left_index_1_joint", "inspire_left_middle_1_joint",
    "inspire_left_ring_1_joint", "inspire_left_little_1_joint",
)


@configclass
class WrenchRefactorRewardsCfg:
    target_reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    target_goal_distance = RewTerm(
        func=goal_distance_v2, weight=5.0, params={"distance_scale": 0.05}
    )
    target_goal_hold = RewTerm(
        func=goal_hold_v2,
        weight=10.0,
        params={"full_reward_radius": 0.025, "outer_radius": 0.035},
    )
    target_contact = RewTerm(
        func=RefactorContactMaintenance,
        weight=0.1,
        params={"threshold_n": 0.01},
    )
    robot_action_rate_penalty = RewTerm(func=RefactorActionRatePenalty, weight=-1.0)
    robot_joint_velocity_penalty = RewTerm(
        func=joint_velocity_penalty,
        weight=-1.0,
        params={
            "arm_cfg": arm_entity(),
            "hand_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_DRIVE_JOINT_NAMES), preserve_order=True
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
    shelf_penalty = RewTerm(
        func=ShelfPenalty,
        weight=-1.0,
        params={"termination_term_name": "shelf_failure"},
    )
    obstacle_contact_penalty = RewTerm(
        func=obstacle_contact_force_cost,
        weight=-0.05,
        params={"free_force": 30.0, "max_cost": 100.0},
    )
    target_goal_first_entry_bonus = RewTerm(func=RefactorFirstEntry, weight=15.0)
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
