"""Explicit Final v0 terms, frozen from Holding OSC ObjectRandom v3."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter import shelf_failure
from ..reward_functions.contact_sweep_clutter_fusion import ObjectAccelerationPenalty
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import active_object_tilt_cost
from ..reward_functions.contact_sweep_clutter_fusion import active_obstacle_displacement_cost
from ..reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
from ..reward_functions.contact_sweep_clutter_fusion import eef_linear_speed_cost
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_joint_speed_cost_limit
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import hand_contact_or_goal
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import goal_settling_centered
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import ReachingOrGoal


@configclass
class FinalRewardsCfg:
    reaching = RewTerm(
        func=ReachingOrGoal,
        weight=30.0,
        params={'near_radius': 0.1, 'frame_name': 'ee_frame', 'goal_near_radius': 1.0},
    )
    goal_distance = RewTerm(func=DexPlanarGoalProgress, weight=100.0, params={'position_tolerance': 0.0})
    goal_hold = RewTerm(
        func=goal_settling_centered,
        weight=4.0,
        params={'position_tolerance': 0.02, 'speed_scale': 0.05},
    )
    hand_contact = RewTerm(func=hand_contact_or_goal, weight=1.45, params={'threshold_n': 0.01})
    arm_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'arm', 'max_cost': 100.0})
    hand_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'hand', 'max_cost': 100.0})
    arm_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.03, params={'asset_cfg': arm_entity(), 'max_cost': 100.0})
    hand_joint_velocity = RewTerm(
        func=joint_vel_l2,
        weight=-0.01,
        params={
            'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True),
            'max_cost': 100.0,
        },
    )
    object = RewTerm(
        func=combined_active_object_cost,
        weight=-30.0,
        params={
            'free_angle': math.radians(5.0),
            'fail_angle': 0.9,
            'displacement_ratio': 5.0,
            'displacement_max_cost': 0.04,
        },
    )
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(
        func=exponential_pushing_speed_cost_limit,
        weight=-5.0,
        params={'free_speed': 0.1, 'max_cost': 100.0},
    )
    arm_speed_excess = RewTerm(
        func=exponential_joint_speed_cost_limit,
        weight=-1.0,
        params={'asset_cfg': arm_entity(), 'free_speed': 1.0, 'max_cost': 100.0},
    )
    shelf_failure = RewTerm(func=shelf_failure, weight=-3.0)
    eef_linear_speed = RewTerm(func=eef_linear_speed_cost, weight=-1.0, params={'free_speed': 1.5, 'max_cost': 100.0})
    eef_orientation = None
    obstacle_contact = RewTerm(
        func=obstacle_contact_force_cost,
        weight=0.0,
        params={'free_force': 80.0, 'max_cost': 100.0},
    )
    object_acceleration = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=0.0,
        params={'free_acceleration': 50.0, 'max_cost': 100.0},
    )
    wrench_change = RewTerm(
        func=WrenchMagnitudeChangePenalty,
        weight=0.0,
        params={
            'free_force_change': 50.0,
            'free_moment_change': 5.0,
            'saturation_ratio': 2.5,
            'sharpness': 3.0,
            'max_positive_fraction': 0.05,
        },
    )


from ..reward_functions.final import action_rate_penalty, joint_velocity_penalty, ObjectPenalty, ShelfPenalty
from ..reward_functions.final import exponential_planar_goal_proximity
from ..reward_functions.final import strict_goal_settling_centered, strict_hand_contact_or_goal


@configclass
class FinalV1RewardsCfg:
    """Negative outer weights keep wrench's positive-reward budget unchanged."""

    hand_reaching = FinalRewardsCfg().reaching
    goal_distance = FinalRewardsCfg().goal_distance
    goal_hold = FinalRewardsCfg().goal_hold
    hand_contact = FinalRewardsCfg().hand_contact
    action_rate_penalty = RewTerm(func=action_rate_penalty, weight=-1.0)
    joint_velocity_penalty = RewTerm(
        func=joint_velocity_penalty, weight=-1.0,
        params={'arm_cfg': arm_entity(), 'hand_cfg': SceneEntityCfg(
            'robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True)},
    )
    object_penalty = RewTerm(
        func=ObjectPenalty, weight=-1.0,
        params={**FinalRewardsCfg().object.params, 'free_acceleration': 50.0,
                'acceleration_weight': 0.0, 'max_cost': 100.0},
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
        params={'distance_scale': 0.05},
    )


@configclass
class FinalV1TestRewardsCfg(FinalV1RewardsCfg):
    """Expose the three v1 object costs separately without changing their sum."""

    object_penalty = None
    object_tilt_penalty = RewTerm(
        func=active_object_tilt_cost, weight=-30.0,
        params={'free_angle': math.radians(5.0), 'fail_angle': 0.9},
    )
    object_displacement_penalty = RewTerm(
        func=active_obstacle_displacement_cost, weight=-150.0,
        params={'displacement_max_cost': 0.04},
    )
    object_acceleration_penalty = RewTerm(
        func=ObjectAccelerationPenalty, weight=0.0,
        params={'free_acceleration': 50.0, 'max_cost': 100.0},
    )


@configclass
class FinalGoalAccuracyRewardsCfg(FinalV1RewardsCfg):
    """Increase precise hold credit and offset the fixed, wider speed allowance."""

    hand_reaching = RewTerm(
        func=ReachingOrGoal, weight=30.0,
        params={'near_radius': 0.01, 'frame_name': 'ee_frame', 'goal_near_radius': 0.01},
    )
    goal_hold = RewTerm(
        func=strict_goal_settling_centered, weight=8.0,
        params={'position_tolerance': 0.01, 'speed_scale': 0.05},
    )
    hand_contact = RewTerm(func=strict_hand_contact_or_goal, weight=1.45,
                           params={'threshold_n': 0.01})
    object_speed_penalty = None
    target_speed_penalty = RewTerm(
        func=exponential_pushing_speed_cost_limit, weight=-10.0,
        params={'free_speed': 0.1, 'max_cost': 100.0},
    )
