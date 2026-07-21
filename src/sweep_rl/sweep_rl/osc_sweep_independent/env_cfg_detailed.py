"""Detailed-reward variant inheriting the independent shelf sweep task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import (
    ARM_CFG,
    EEF_CFG,
    FT_CFG,
    PAD_SENSORS,
    ROBOT_CONTACT_SENSORS,
    TARGET_CFG,
    UR5eOscSweepIndependentEnvCfg,
)


@configclass
class DetailedRewardsCfg:
    """Separated Reach, Sweep, Home, and safety learning signals."""

    reach_pose_tracking = RewTerm(
        func=mdp.reach_pose_tracking,
        weight=4.0,
        params={
            "command_name": "desired_motion",
            "distance_std": 0.12,
            "surface_clearance": 0.008,
            "table_side_pad_offset": 0.055,
            "eef_cfg": EEF_CFG,
            "object_cfg": TARGET_CFG,
        },
    )
    reach_pose_error = RewTerm(
        func=mdp.reach_pose_error,
        weight=-1.0,
        params={
            "command_name": "desired_motion",
            "distance_scale": 0.12,
            "maximum_normalized_error": 3.0,
            "surface_clearance": 0.008,
            "table_side_pad_offset": 0.055,
            "eef_cfg": EEF_CFG,
            "object_cfg": TARGET_CFG,
        },
    )

    sweep_contact = RewTerm(
        func=mdp.sweep_contact,
        weight=1.5,
        params={
            "command_name": "desired_motion",
            "sensor_names": PAD_SENSORS,
            "force_threshold": 0.25,
        },
    )
    sweep_velocity_tracking = RewTerm(
        func=mdp.sweep_velocity_tracking,
        weight=8.0,
        params={
            "command_name": "desired_motion",
            "velocity_std": 0.035,
            "acceleration_distance": 0.030,
            "stopping_distance": 0.050,
            "initial_speed_fraction": 0.25,
            "sensor_names": PAD_SENSORS,
            "force_threshold": 0.25,
            "object_cfg": TARGET_CFG,
        },
    )
    sweep_forward_progress = RewTerm(
        func=mdp.sweep_forward_progress,
        weight=2.0,
        params={"command_name": "desired_motion", "object_cfg": TARGET_CFG},
    )
    sweep_endpoint_error = RewTerm(
        func=mdp.sweep_endpoint_error,
        weight=-4.0,
        params={
            "command_name": "desired_motion",
            "maximum_normalized_error": 2.0,
            "object_cfg": TARGET_CFG,
        },
    )
    sweep_lateral_error = RewTerm(
        func=mdp.sweep_lateral_error,
        weight=-3.0,
        params={"command_name": "desired_motion"},
    )
    sweep_overshoot = RewTerm(
        func=mdp.sweep_overshoot,
        weight=-6.0,
        params={"command_name": "desired_motion"},
    )
    sweep_stopped_at_goal = RewTerm(
        func=mdp.sweep_stopped_at_goal,
        weight=15.0,
        params={
            "command_name": "desired_motion",
            "position_std": 0.030,
            "speed_std": 0.020,
            "object_cfg": TARGET_CFG,
        },
    )

    home_joint_pose = RewTerm(
        func=mdp.home_joint_pose_tracking,
        weight=12.0,
        params={
            "command_name": "desired_motion",
            "joint_std": 0.35,
            "asset_cfg": ARM_CFG,
        },
    )
    home_joint_error = RewTerm(
        func=mdp.home_joint_error,
        weight=-2.0,
        params={
            "command_name": "desired_motion",
            "error_scale": 0.75,
            "maximum_normalized_error": 3.0,
            "asset_cfg": ARM_CFG,
        },
    )
    home_clearance = RewTerm(
        func=mdp.home_eef_clearance,
        weight=2.0,
        params={
            "command_name": "desired_motion",
            "safe_distance": 0.22,
            "eef_cfg": EEF_CFG,
            "object_cfg": TARGET_CFG,
        },
    )
    post_goal_contact = RewTerm(
        func=mdp.home_contact_penalty,
        weight=-10.0,
        params={
            "command_name": "desired_motion",
            "sensor_names": ROBOT_CONTACT_SENSORS,
            "force_threshold": 0.25,
        },
    )
    goal_hold_error = RewTerm(
        func=mdp.home_goal_hold_error,
        weight=-8.0,
        params={
            "command_name": "desired_motion",
            "maximum_normalized_error": 2.0,
            "object_cfg": TARGET_CFG,
        },
    )
    post_goal_object_speed = RewTerm(
        func=mdp.home_object_speed,
        weight=-2.0,
        params={
            "command_name": "desired_motion",
            "speed_scale": 0.05,
            "maximum_normalized_speed": 4.0,
            "object_cfg": TARGET_CFG,
        },
    )
    post_goal_object_displacement = RewTerm(
        func=mdp.home_object_displacement,
        weight=-6.0,
        params={
            "command_name": "desired_motion",
            "displacement_scale": 0.010,
            "maximum_normalized_displacement": 4.0,
            "object_cfg": TARGET_CFG,
        },
    )
    home_time = RewTerm(
        func=mdp.home_time,
        weight=-0.3,
        params={"command_name": "desired_motion"},
    )
    home_success = RewTerm(
        func=mdp.home_success_bonus,
        weight=30.0,
        params={
            "command_name": "desired_motion",
            "joint_position_threshold": 0.12,
            "joint_speed_threshold": 0.15,
            "endpoint_threshold": 0.030,
            "object_speed_threshold": 0.025,
            "object_displacement_threshold": 0.010,
            "sensor_names": ROBOT_CONTACT_SENSORS,
            "contact_force_threshold": 0.25,
            "asset_cfg": ARM_CFG,
            "object_cfg": TARGET_CFG,
        },
    )

    ft_torque = RewTerm(
        func=mdp.ft_torque_excess,
        weight=-0.02,
        params={"threshold": 1.5, "asset_cfg": FT_CFG},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": ARM_CFG},
    )
    commanded_effort = RewTerm(
        func=mdp.commanded_joint_effort_l2,
        weight=-0.01,
        params={"action_name": "arm_action"},
    )
    torque_saturation = RewTerm(
        func=mdp.torque_saturation,
        weight=-0.5,
        params={"action_name": "arm_action"},
    )
    failure_termination = RewTerm(
        func=mdp.failure_termination_penalty,
        weight=-5.0,
        params={
            "term_names": (
                "contact_lost",
                "object_inside_gripper",
                "home_contact",
                "object_disturbed_home",
                "target_invalid_pose",
                "excessive_wrench",
                "arm_speed",
                "shelf_collision",
                "self_collision",
            ),
            "minimum_remaining_time": 1.0,
        },
    )


@configclass
class UR5eOscSweepIndependentDetailedEnvCfg(UR5eOscSweepIndependentEnvCfg):
    """Inherited shelf task with decomposed, phase-specific shaping."""

    rewards: DetailedRewardsCfg = DetailedRewardsCfg()
