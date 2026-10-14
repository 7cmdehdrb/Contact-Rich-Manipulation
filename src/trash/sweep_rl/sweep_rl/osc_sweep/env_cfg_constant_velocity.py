"""Force-free UR5e sweep task with a constant object-velocity objective."""

from __future__ import annotations

import math

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .assets import (
    ARM_JOINT_NAMES,
    CONTACT_PAD_SIZE,
    EEF_CENTER_BODY_NAME,
    GRIPPER_SIDE_FACE_NORMAL_AXIS,
)
from .env_cfg import (
    ARM_ENTITY_CFG,
    EEF_ENTITY_CFG,
    FT_ENTITY_CFG,
    UR5eOscSweepEnvCfg,
)


@configclass
class ConstantVelocityCommandsCfg:
    """Force-free motion command with one fixed cruise-speed target."""

    desired_motion = mdp.ConstantVelocitySweepCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.10, 0.22),
        target_speed_range=(0.08, 0.08),
        debug_vis=False,
    )


@configclass
class ConstantVelocityActionsCfg:
    """Variable-stiffness OSC arm control with a non-policy open gripper."""

    arm_action = mdp.OpenGripperSweepOperationalSpaceActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=EEF_CENTER_BODY_NAME,
        body_offset=None,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="variable_kp",
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=(
                120.0,
                120.0,
                120.0,
                35.0,
                35.0,
                35.0,
            ),
            motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            motion_stiffness_limits_task=(20.0, 300.0),
            nullspace_control="none",
        ),
        position_scale=0.025,
        orientation_scale=0.12,
        stiffness_scale=1.0,
        nullspace_joint_pos_target="none",
        effort_limit_scale=0.9,
        gripper_joint_names=[".*(finger|knuckle).*"],
        gripper_open_position=0.0,
    )


@configclass
class ConstantVelocityObservationsCfg:
    """55-D state without target-contact force or contact-point signals."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": ARM_ENTITY_CFG},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": ARM_ENTITY_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": ARM_ENTITY_CFG},
        )
        eef_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={"asset_cfg": EEF_ENTITY_CFG},
        )
        initial_target_pose = ObsTerm(
            func=mdp.initial_target_pose_b,
            params={"command_name": "desired_motion"},
        )
        current_target_pose = ObsTerm(
            func=mdp.current_target_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("target_object"),
            },
        )
        object_linear_velocity = ObsTerm(
            func=mdp.object_linear_velocity_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("target_object"),
            },
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        desired_motion = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "desired_motion"},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ConstantVelocityRewardsCfg:
    """Reward accel/cruise/stop velocity tracking and final placement."""

    push_pose_error = RewTerm(
        func=mdp.current_precontact_pose_error,
        weight=-0.35,
        params={
            "command_name": "desired_motion",
            "distance_scale": 0.10,
            "stand_off": 0.065,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    side_direction_error = RewTerm(
        func=mdp.gripper_side_direction_error,
        weight=-0.25,
        params={
            "command_name": "desired_motion",
            "side_axis_local": (1.0, 0.0, 0.0),
            "proximity_std": 0.20,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    target_contact = RewTerm(
        func=mdp.target_contact_bonus,
        weight=0.50,
        params={"sensor_names": ("left_contact", "right_contact")},
    )
    side_center_contact = RewTerm(
        func=mdp.side_pad_center_contact,
        weight=0.75,
        params={
            "sensor_names": ("left_contact", "right_contact"),
            "pad_size": CONTACT_PAD_SIZE,
            "face_normal_axis": GRIPPER_SIDE_FACE_NORMAL_AXIS,
            "center_sigma": 0.45,
            "face_sigma": 0.25,
        },
    )
    contact_forward_progress = RewTerm(
        func=mdp.contact_forward_progress,
        weight=3.0,
        params={
            "command_name": "desired_motion",
            "acceleration_distance": 0.025,
            "stopping_distance": 0.04,
            "initial_speed_fraction": 0.25,
            "endpoint_threshold": 0.020,
            "maximum_normalized_speed": 1.25,
            "sensor_names": ("left_contact", "right_contact"),
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    velocity_tracking = RewTerm(
        func=mdp.object_velocity_profile_tracking,
        weight=10.0,
        params={
            "command_name": "desired_motion",
            "std": 0.025,
            "acceleration_distance": 0.025,
            "stopping_distance": 0.04,
            "initial_speed_fraction": 0.25,
            "endpoint_threshold": 0.020,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    endpoint_error = RewTerm(
        func=mdp.normalized_endpoint_error,
        weight=-5.0,
        params={
            "command_name": "desired_motion",
            "maximum_error": 2.0,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    stopped_at_goal = RewTerm(
        func=mdp.stopped_at_goal_reward,
        weight=20.0,
        params={
            "command_name": "desired_motion",
            "position_std": 0.025,
            "speed_std": 0.020,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    success = RewTerm(
        func=mdp.stopped_sweep_success_bonus,
        weight=40.0,
        params={
            "command_name": "desired_motion",
            "endpoint_threshold": 0.020,
            "lateral_threshold": 0.10,
            "speed_threshold": 0.020,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    failure_termination = RewTerm(
        func=mdp.remaining_horizon_failure_penalty,
        weight=-8.0,
        params={
            "term_names": (
                "target_invalid_pose",
                "excessive_wrench",
                "arm_speed",
            ),
            "minimum_penalty_time": 1.0,
        },
    )

    lateral_error = RewTerm(
        func=mdp.lateral_displacement,
        weight=-3.0,
        params={"command_name": "desired_motion"},
    )
    overshoot = RewTerm(
        func=mdp.overshoot_penalty,
        weight=-8.0,
        params={"command_name": "desired_motion"},
    )
    stall = RewTerm(
        func=mdp.object_stall_penalty,
        weight=-6.0,
        params={
            "command_name": "desired_motion",
            "startup_grace_time": 0.40,
            "endpoint_threshold": 0.020,
            "minimum_speed_fraction": 0.50,
            "acceleration_distance": 0.025,
            "stopping_distance": 0.04,
            "initial_speed_fraction": 0.25,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    object_acceleration = RewTerm(
        func=mdp.object_linear_acceleration_penalty,
        weight=-0.15,
        params={
            "acceleration_scale": 1.0,
            "maximum_normalized_penalty": 25.0,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    ft_torque = RewTerm(
        func=mdp.ft_torque_excess,
        weight=-0.02,
        params={"threshold": 1.5, "asset_cfg": FT_ENTITY_CFG},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": ARM_ENTITY_CFG},
    )
    commanded_effort = RewTerm(
        func=mdp.commanded_joint_effort_l2,
        weight=-0.03,
        params={"action_name": "arm_action"},
    )
    torque_saturation = RewTerm(
        func=mdp.torque_saturation,
        weight=-0.5,
        params={"action_name": "arm_action"},
    )


@configclass
class ConstantVelocityTerminationsCfg:
    """Safety failures plus success after a stable endpoint dwell."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.TargetStoppedAtGoal,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "endpoint_threshold": 0.020,
            "lateral_threshold": 0.10,
            "speed_threshold": 0.020,
            "dwell_time": 0.30,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    target_invalid_pose = DoneTerm(
        func=mdp.target_invalid_pose,
        time_out=False,
        params={
            "minimum_height": 0.72,
            "maximum_tilt": 0.80,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    excessive_wrench = DoneTerm(
        func=mdp.excessive_ft_wrench,
        time_out=False,
        params={
            "force_limit": 100.0,
            "torque_limit": 15.0,
            "asset_cfg": FT_ENTITY_CFG,
        },
    )
    arm_speed = DoneTerm(
        func=mdp.arm_joint_speed_limit,
        time_out=False,
        params={"maximum_speed": 6.5, "asset_cfg": ARM_ENTITY_CFG},
    )


@configclass
class UR5eOscSweepConstantVelocityEnvCfg(UR5eOscSweepEnvCfg):
    """Direct baseline-derived task for constant-velocity object sweeping."""

    commands: ConstantVelocityCommandsCfg = ConstantVelocityCommandsCfg()
    actions: ConstantVelocityActionsCfg = ConstantVelocityActionsCfg()
    observations: ConstantVelocityObservationsCfg = ConstantVelocityObservationsCfg()
    rewards: ConstantVelocityRewardsCfg = ConstantVelocityRewardsCfg()
    terminations: ConstantVelocityTerminationsCfg = ConstantVelocityTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The fixed open target is re-applied by the action term every physics
        # step; these gains make the implicit gripper actuator hold that target.
        gripper_actuator = self.scene.robot.actuators["gripper"]
        gripper_actuator.stiffness = 2000.0
        gripper_actuator.damping = 100.0
