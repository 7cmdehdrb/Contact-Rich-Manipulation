"""Holding OSC v2 rewards, fixed 3 cm hold radius; no IK reward terms."""
from ..reward_functions.contact_sweep_clutter_fusion import active_object_tilt_cost, active_obstacle_displacement_cost
from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter import shelf_failure
from ..reward_functions.contact_sweep_clutter_fusion import ObjectAccelerationPenalty
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
from ..reward_functions.contact_sweep_clutter_fusion import eef_linear_speed_cost
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math
from ..reward_functions.contact_sweep_clutter_fusion_holding_scheduled import goal_settling, hand_contact_or_goal
from ..reward_functions.wrench_refactor_relative_ik_test import holding_shelf_failure

@configclass
class WrenchRefactorRelativeIKV0_2TestRewardsCfg:
    reaching = RewTerm(func=DexReachProgress, weight=30.0, params={'near_radius': 0.1, 'frame_name': 'ee_frame'})
    goal_distance = RewTerm(func=DexPlanarGoalProgress, weight=100.0, params={'position_tolerance': 0.0})
    goal_hold = RewTerm(func=goal_settling, weight=4.0, params={'position_tolerance': 0.03, 'speed_scale': 0.05})
    hand_contact = RewTerm(func=hand_contact_or_goal, weight=1.45, params={'threshold_n': 0.01})
    arm_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'arm', 'max_cost': 100.0})
    hand_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'hand', 'max_cost': 100.0})
    arm_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.03, params={'asset_cfg': arm_entity(), 'max_cost': 100.0})
    hand_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.01, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True), 'max_cost': 100.0})
    object = RewTerm(func=combined_active_object_cost, weight=-30.0, params={'free_angle': math.radians(5.0), 'fail_angle': 0.9, 'displacement_ratio': 5.0, 'displacement_max_cost': 0.04})
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = RewTerm(func=exponential_pushing_speed_cost_limit, weight=-5.0, params={'free_speed': 0.1, 'max_cost': 100.0})
    shelf_failure = RewTerm(func=holding_shelf_failure, weight=-3.0)
    eef_linear_speed = RewTerm(func=eef_linear_speed_cost, weight=-1.0, params={'free_speed': 1.5, 'max_cost': 100.0})
    eef_orientation = None
    obstacle_contact = RewTerm(func=obstacle_contact_force_cost, weight=0.0, params={'free_force': 80.0, 'max_cost': 100.0})
    object_acceleration = RewTerm(func=ObjectAccelerationPenalty, weight=0.0, params={'free_acceleration': 50.0, 'max_cost': 100.0})
    wrench_change = RewTerm(func=WrenchMagnitudeChangePenalty, weight=0.0, params={'free_force_change': 50.0, 'free_moment_change': 5.0, 'saturation_ratio': 2.5, 'sharpness': 3.0, 'max_positive_fraction': 0.05})



@configclass
class WrenchRefactorRelativeIKV0_2_1TestRewardsCfg(WrenchRefactorRelativeIKV0_2TestRewardsCfg):
    object = None
    object_tilt = RewTerm(func=active_object_tilt_cost, weight=-30.0,
                          params={'free_angle': math.radians(5.0), 'fail_angle': 0.9})
    obstacle_displacement = RewTerm(func=active_obstacle_displacement_cost, weight=-150.0,
                                    params={'displacement_max_cost': 0.04})
    eef_linear_speed = None
    pushing_speed = None
