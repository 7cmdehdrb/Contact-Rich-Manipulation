from ..reward_functions.stabilize_sweep import squeeze,stage,initial_movement,initial_rotation,sweep_progress,sweep_retreat
from ..reward_functions.contact_sweep_clutter_fusion import active_object_tilt_cost, active_obstacle_displacement_cost
from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math
from ..reward_functions.wrench_refactor_relative_ik_test import holding_shelf_failure
from ..reward_functions.sweep_simple import finger_action_rate_l2
from .sweep_simple import SweepSimpleFingerRewardsCfg

@configclass
class StabilizeSweepRewardsCfg:
    stage = RewTerm(func=stage,weight=1.0)
    initial_movement = RewTerm(func=initial_movement,weight=-1.0)
    initial_rotation = RewTerm(func=initial_rotation,weight=-1.0)
    squeeze = RewTerm(func=squeeze,weight=-1.0)
    sweep_progress = RewTerm(func=sweep_progress,weight=30.0)
    sweep_retreat = RewTerm(func=sweep_retreat,weight=-38.91)
    reaching = None
    goal_distance = None
    goal_hold = None
    hand_contact = None
    arm_action_rate = RewTerm(func=finger_action_rate_l2, weight=-0.03, params={'action_group': 'arm'})
    hand_action_rate = RewTerm(func=finger_action_rate_l2, weight=-0.03, params={'action_group': 'hand'})
    arm_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.03, params={'asset_cfg': arm_entity(), 'max_cost': 100.0})
    hand_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.01, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True), 'max_cost': 100.0})
    object = None
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    pushing_speed = None
    shelf_failure = RewTerm(func=holding_shelf_failure, weight=-3.0)
    eef_linear_speed = None
    eef_orientation = None
    obstacle_contact = RewTerm(func=obstacle_contact_force_cost, weight=0.0, params={'free_force': 80.0, 'max_cost': 100.0})
    object_acceleration = None
    wrench_change = RewTerm(func=WrenchMagnitudeChangePenalty, weight=0.0, params={'free_force_change': 50.0, 'free_moment_change': 5.0, 'saturation_ratio': 2.5, 'sharpness': 3.0, 'max_positive_fraction': None})
    object_tilt = RewTerm(func=active_object_tilt_cost, weight=-30.0, params={'free_angle': math.radians(5.0), 'fail_angle': 0.9})
    obstacle_displacement = RewTerm(func=active_obstacle_displacement_cost, weight=-150.0, params={'displacement_max_cost': 0.04})
    target_inverse_distance = None


@configclass
class SweepOnlyRewardsCfg(StabilizeSweepRewardsCfg):
    stage = None
    initial_movement = None
    initial_rotation = None
    squeeze = None
    sweep_retreat = StabilizeSweepRewardsCfg().sweep_retreat.replace(weight=-20.0)


@configclass
class StabilizeSweepTestRewardsCfg(StabilizeSweepRewardsCfg):
    reaching = SweepSimpleFingerRewardsCfg().reaching.copy()


@configclass
class SweepOnlyTestRewardsCfg(SweepOnlyRewardsCfg):
    reaching = SweepSimpleFingerRewardsCfg().reaching.copy()
