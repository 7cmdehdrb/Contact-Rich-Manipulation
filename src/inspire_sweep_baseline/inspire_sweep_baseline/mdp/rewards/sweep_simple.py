"""Simple family: pinned effective v0_2_2_4 configuration, without historical Cfg inheritance."""
from ..reward_functions.sweep_simple import AreaMatchedLogGoalProgress, RetreatReach, RetreatContactMaintenance
from ..reward_functions.sweep_simple import ContactLogGoalProgress, semi_prof_force_direction, cosine_force_direction
from ..reward_functions.contact_sweep_clutter_fusion import active_object_tilt_cost, active_obstacle_displacement_cost
from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter import shelf_failure
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math
from ..reward_functions.wrench_refactor_relative_ik_test import holding_shelf_failure
from ..reward_functions.wrench_refactor import RelativeInverseDistance
from ..reward_functions.sweep_simple import GoalContactMaintenance, finger_action_rate_l2
from ..reward_functions.wrench_refactor_relative_ik_test import ContactConditionedSweep
from ..reward_functions.wrench_refactor_relative_ik_test import InverseDistanceReach

@configclass
class SweepSimpleRewardsCfg:
    reaching = RewTerm(func=DexReachProgress, weight=30.0, params={'near_radius': 0.1, 'frame_name': 'ee_frame'})
    goal_distance = RewTerm(func=DexPlanarGoalProgress, weight=100.0, params={'position_tolerance': 0.0})
    goal_hold = None
    hand_contact = RewTerm(func=GoalContactMaintenance, weight=0.1, params={'threshold_n': 0.01, 'eef_height_limit': 0.12})
    arm_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'arm'})
    hand_action_rate = RewTerm(func=action_rate_l2, weight=-0.03, params={'action_group': 'hand'})
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
    wrench_change = RewTerm(func=WrenchMagnitudeChangePenalty, weight=0.0, params={'free_force_change': 50.0, 'free_moment_change': 5.0, 'saturation_ratio': 2.5, 'sharpness': 3.0, 'max_positive_fraction': 0.05})
    object_tilt = RewTerm(func=active_object_tilt_cost, weight=-30.0, params={'free_angle': math.radians(5.0), 'fail_angle': 0.9})
    obstacle_displacement = RewTerm(func=active_obstacle_displacement_cost, weight=-150.0, params={'displacement_max_cost': 0.04})
    target_inverse_distance = RewTerm(func=RelativeInverseDistance, weight=-25.94)

@configclass
class SweepSimpleProfRewardsCfg(SweepSimpleRewardsCfg):
    goal_distance = RewTerm(func=ContactConditionedSweep, weight=120.0, params={'component': 'approach', 'threshold_n': 0.01, 'eef_height_limit': 0.12})
    direction_deviation = RewTerm(func=ContactConditionedSweep, weight=-6.0, params={'component': 'direction', 'threshold_n': 0.01, 'eef_height_limit': 0.12})
    hand_contact = None

@configclass
class SweepSimpleV1RewardsCfg(SweepSimpleRewardsCfg):
    reaching = RewTerm(func=InverseDistanceReach, weight=30.0, params={'near_radius': 0.1, 'distance_scale': 0.1, 'frame_name': 'ee_frame'})

@configclass
class SweepSimpleProfV1RewardsCfg(SweepSimpleProfRewardsCfg):
    reaching = RewTerm(func=InverseDistanceReach, weight=30.0, params={'near_radius': 0.1, 'distance_scale': 0.1, 'frame_name': 'ee_frame'})


@configclass
class SweepSimpleProfV2RewardsCfg(SweepSimpleProfV1RewardsCfg):
    def __post_init__(self):
        self.goal_distance.params['eef_height_limit'] = .12
        self.direction_deviation.params['eef_height_limit'] = .12
        self.arm_action_rate.func = finger_action_rate_l2
        self.hand_action_rate.func = finger_action_rate_l2


@configclass
class SweepSimpleFingerRewardsCfg(SweepSimpleV1RewardsCfg):
    def __post_init__(self):
        self.arm_action_rate.func = finger_action_rate_l2
        self.hand_action_rate.func = finger_action_rate_l2


@configclass
class SweepSimpleFingerForceDirectionRewardsCfg(SweepSimpleFingerRewardsCfg):
    force_direction = RewTerm(func=cosine_force_direction, weight=.33, params={
        'min_force': 1., 'near_radius': .01})

    def __post_init__(self):
        super().__post_init__()
        self.hand_contact.params['goal_radius'] = .01
        # The inherited Wrench budget requires all positive terms before it.
        # Insert only the new term; preserve every parent's relative term order.
        terms = list(vars(self).items())
        direction = self.force_direction
        self.__dict__.clear()
        for name, term in terms:
            if name == 'wrench_change':
                self.force_direction = direction
            if name != 'force_direction':
                setattr(self, name, term)


@configclass
class SweepSimpleFingerV1RewardsCfg(SweepSimpleFingerRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.arm_joint_velocity.weight = -.03


@configclass
class SweepSimpleFingerV2RewardsCfg(SweepSimpleFingerV1RewardsCfg):
    goal_distance = RewTerm(func=AreaMatchedLogGoalProgress, weight=100., params={'epsilon': .01})
    reaching = RewTerm(func=RetreatReach, weight=30., params={
        'near_radius': .1, 'distance_scale': .1, 'frame_name': 'ee_frame', 'retreat_radius': .02})
    hand_contact = RewTerm(func=RetreatContactMaintenance, weight=.1, params={
        'threshold_n': .01, 'eef_height_limit': .12})


@configclass
class SweepSimpleSemiProfRewardsCfg(SweepSimpleFingerV1RewardsCfg):
    goal_distance = RewTerm(func=ContactLogGoalProgress, weight=120., params={
        'epsilon': .01, 'threshold_n': .01, 'eef_height_limit': .12})
    hand_contact = None
    target_inverse_distance = RewTerm(func=RelativeInverseDistance, weight=-31.128)
    force_direction = RewTerm(func=semi_prof_force_direction, weight=.33, params={
        'min_force': 1., 'near_radius': .01})

    def __post_init__(self):
        super().__post_init__()
        # Reward weights express relative importance, not a hard reward budget.
        self.wrench_change.params['max_positive_fraction'] = None


@configclass
class SweepSimpleFingerRawV1RewardsCfg(SweepSimpleFingerV1RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.arm_joint_velocity.weight = -.03


@configclass
class SweepSimpleFingerRawV2RewardsCfg(SweepSimpleFingerV2RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.arm_joint_velocity.weight = -.03


@configclass
class SweepSimpleSemiProfRawRewardsCfg(SweepSimpleSemiProfRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.arm_joint_velocity.weight = -.03
