"""Explicit fused rewards; weights apply once through RewardManager and dt.

Explicit reward contract for the v1.0 branch; weights are rates per second.

Explicit reward contract for the v1_1 branch; weights are rates per second.

Explicit reward contract for the v1_1_1 branch; weights are rates per second.

Explicit reward contract for the v1_1_2 branch; weights are rates per second.

Explicit reward contract for the v1_2 branch; weights are rates per second.

Explicit reward contract for the v1_3 branch; weights are rates per second.

Explicit reward contract for the v1_3_1 branch; weights are rates per second.

Inherit v1.0 penalties and increase only the requested positive weights."""

from ...assets.robot import HAND_JOINT_NAMES
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..reward_functions.contact_sweep_clutter import contact_any
from ..reward_functions.contact_sweep_clutter import contact_count
from ..reward_functions.contact_sweep_clutter import goal_hold_xy
from ..reward_functions.contact_sweep_clutter import shelf_failure
from ..reward_functions.contact_sweep_clutter_fusion import ObjectAccelerationPenalty
from ..reward_functions.contact_sweep_clutter_fusion import WrenchMagnitudeChangePenalty
from ..reward_functions.contact_sweep_clutter_fusion import WrenchRatePenalty
from ..reward_functions.contact_sweep_clutter_fusion import combined_action_rate
from ..reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
from ..reward_functions.contact_sweep_clutter_fusion import combined_joint_velocity
from ..reward_functions.contact_sweep_clutter_fusion import combined_object_cost
from ..reward_functions.contact_sweep_clutter_fusion import eef_linear_speed_cost
from ..reward_functions.contact_sweep_clutter_fusion import eef_upward_z_orientation_cost
from ..reward_functions.contact_sweep_clutter_fusion import hand_contact_any
from ..reward_functions.contact_sweep_clutter_fusion import obstacle_contact_force_cost
from ..reward_functions.contact_sweep_clutter_fusion import scheduled_goal_hold_xy
from ..reward_functions.dex_rewards import DexPlanarGoalProgress
from ..reward_functions.dex_rewards import DexReachProgress
from ..reward_functions.goal_rewards import action_rate_l2
from ..reward_functions.goal_rewards import exponential_joint_speed_cost_limit
from ..reward_functions.goal_rewards import exponential_pushing_speed_cost_limit
from ..reward_functions.goal_rewards import joint_vel_l2
from ..reward_functions.sweep import ShelfCollisionPenalty
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math as math

@configclass
class ddContactSweepClutterFusionRewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
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


@configclass
class ContactSweepClutterFusionV1_0RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    # 긴급 조정 : Goal Hold를 예전과 동일하게 변경
    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.
    # Alpha fades to zero within 1.5 cm outside the live goal-hold tolerance.

    # 긴급 조정 : Fade width를 0.0으로 변경 -> 감쇄 없음
    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.0},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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

    # 긴급 조정 : Obstacle Contact Free Force를 대폭 상향 조정
    obstacle_contact = RewTerm(
        func=obstacle_contact_force_cost,
        weight=-1.0,
        params={"free_force": 2000.0, "max_cost": 100.0},
    )
    #  긴급 조정 : Object Acceleration Free Acceleration를 대폭 상향 조정
    object_acceleration = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=-1.0,
        params={"free_acceleration": 100.0, "max_cost": 100.0},
    )

    # Preserve the v3 normalized exponential, absolute cap and positive budget.
    #  긴급 조정 : Wrench Change Free Force Change와 Free Moment Change를 대폭 상향 조정
    wrench_change = RewTerm(
        func=WrenchMagnitudeChangePenalty,
        weight=-0.1,
        params={
            "free_force_change": 5000.0,
            "free_moment_change": 500.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


@configclass
class ContactSweepClutterFusionV1_1RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.

    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.0},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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


@configclass
class ContactSweepClutterFusionV1_1_1RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.

    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.0},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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
        weight=-1.0,
        params={"free_force": 2000.0, "max_cost": 100.0},
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
            "free_force_change": 5000.0,
            "free_moment_change": 500.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


@configclass
class ContactSweepClutterFusionV1_1_2RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.

    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.0},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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
        weight=-1.0,
        params={"free_force": 100.0, "max_cost": 100.0},
    )
    object_acceleration = RewTerm(
        func=ObjectAccelerationPenalty,
        weight=-1.0,
        params={"free_acceleration": 1_000_000.0, "max_cost": 100.0},
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


@configclass
class ContactSweepClutterFusionV1_2RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
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
            "displacement_ratio": 2.5,
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


@configclass
class ContactSweepClutterFusionV1_3RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
        weight=100.0,
        params={"position_tolerance": 0.01},
    )

    goal_hold = RewTerm(
        func=goal_hold_xy, weight=0.9, params={"position_tolerance": 0.01}
    )
    # Late v3 signed return excluding tactile = 12.990838965 per episode.
    # Five seconds at alpha=1 -> 1.45 reward, a 10.04% baseline share.

    hand_contact = RewTerm(
        func=hand_contact_any,
        weight=0.29,
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.0},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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
        weight=-1.0,
        params={"free_force": 200.0, "max_cost": 100.0},
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
            "free_force_change": 1000.0,
            "free_moment_change": 100.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


@configclass
class ContactSweepClutterFusionV1_3_1RewardsCfg:
    reaching = RewTerm(
        func=DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=DexPlanarGoalProgress,
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
        params={"threshold_n": 0.01, "goal_contact_fade_width": 0.001},
    )
    # Workspace contract: separate raw-square costs, fixed coefficients.
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
            "displacement_ratio": 2.5,
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
        weight=-1.0,
        params={"free_force": 200.0, "max_cost": 100.0},
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
            "free_force_change": 1000.0,
            "free_moment_change": 100.0,
            "saturation_ratio": 2.5,
            "sharpness": 3.0,
            "max_positive_fraction": 0.05,
        },
    )


@configclass
class ContactSweepClutterFusionV1_4RewardsCfg(ContactSweepClutterFusionV1_0RewardsCfg):
    goal_hold = ContactSweepClutterFusionV1_0RewardsCfg().goal_hold.replace(
        func=scheduled_goal_hold_xy, weight=9.0, params={"position_tolerance": 0.0},
    )
    hand_contact = ContactSweepClutterFusionV1_0RewardsCfg().hand_contact.replace(weight=1.45)
