from ..reward_functions.stabilize_sweep import sweep_retreat, action_limit_excess
from ..reward_functions.sweep_contact_preference import ContactSideSweepProgress
from ..reward_functions.prof_revision import reaching, contact_formation
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

@configclass
class ProfRevisionRewardsCfg:
    sweep_progress = RewTerm(func=ContactSideSweepProgress, weight=100., params={'preferred_multiplier': 1.5, 'other_multiplier': 0.7})
    sweep_retreat = RewTerm(func=sweep_retreat, weight=-20.)
    reaching = RewTerm(func=reaching, weight=4.9)
    arm_action_rate = RewTerm(func=finger_action_rate_l2, weight=-0.03, params={'action_group': 'arm', 'max_cost': 100.})
    hand_action_rate = RewTerm(func=finger_action_rate_l2, weight=-0.03, params={'action_group': 'hand', 'max_cost': 100.})
    arm_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.03, params={'asset_cfg': arm_entity(), 'max_cost': 100.})
    hand_joint_velocity = RewTerm(func=joint_vel_l2, weight=-0.03, params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(HAND_JOINT_NAMES), preserve_order=True), 'max_cost': 100.})
    shelf_collision = RewTerm(func=ShelfCollisionPenalty, weight=-0.5)
    shelf_failure = RewTerm(func=holding_shelf_failure, weight=-3.)
    obstacle_contact = RewTerm(func=obstacle_contact_force_cost, weight=-0.04, params={'free_force': 50., 'max_cost': 100.})
    wrench_change = RewTerm(func=WrenchMagnitudeChangePenalty, weight=-0.1, params={'free_force_change': 30., 'free_moment_change': 3., 'saturation_ratio': 2.5, 'sharpness': 3., 'max_positive_fraction': None})
    object_tilt = RewTerm(func=active_object_tilt_cost, weight=-30., params={'free_angle': math.radians(5.), 'fail_angle': 0.9})
    obstacle_displacement = RewTerm(func=active_obstacle_displacement_cost, weight=-150., params={'displacement_max_cost': 0.04})
    arm_action_limit = RewTerm(func=action_limit_excess, weight=-3., params={'action_group': 'arm', 'max_cost': 10.})
    hand_action_limit = RewTerm(func=action_limit_excess, weight=-3., params={'action_group': 'hand', 'hand_normalized': True, 'max_cost': 10.})
    contact_formation = RewTerm(func=contact_formation, weight=0.245)
