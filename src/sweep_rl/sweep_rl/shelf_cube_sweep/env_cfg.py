"""Shelf Cube Sweep task inherited from the collision-safe PreReach task."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from sweep_rl.shelf_cube_pre_reach.env_cfg import (
    BEHIND_WIDTH_SCALE,
    CUBE_CENTER_HEIGHT,
    CUBE_DEPTH,
    CUBE_HEIGHT,
    CUBE_MASS,
    CUBE_WIDTH,
    CubePreReachObservationsCfg,
    EE_FRAME_CFG,
    GRIPPER_BASE_BODY_PATH,
    PRE_REACH_Z_OFFSET,
    ROBOT_CONTACT_BODY_PATHS,
    ROBOT_CFG,
    SHELF_CFG,
    SHELF_COLLISION_WEIGHT,
    SHELF_CONTACT_FORCE_THRESHOLD,
    SHELF_FLOOR_SURFACE_TOLERANCE,
    SHELF_FLOOR_X_BOUNDS,
    SHELF_FLOOR_Y_BOUNDS,
    TARGET_OBJECT_CFG,
    TARGET_PITCH,
    TARGET_POS_X,
    TARGET_POS_Y,
    TARGET_POS_Z,
    TARGET_ROLL,
    TARGET_YAW,
    CubePreReachCommandsCfg,
    CubePreReachRewardsCfg,
    CubePreReachRewardsCfgV1,
    ShelfCubePreReachSceneCfg,
    UR5eGripperShelfCubePreReachEnvCfg,
    UR5eGripperShelfCubePreReachEnvCfgV1,
)
from sweep_rl.shelf_cube_pre_reach.mdp.observations import (
    cube_width as cube_width_observation,
)
from sweep_rl.shelf_cube_pre_reach import mdp as cube_pre_reach_mdp
from sweep_rl.shelf_reach.env_cfg import (
    ARM_JOINT_NAMES,
    EventCfg,
    GRIPPER_JOINT_NAMES,
    ROBOT_BASE_BODY_PATH,
    TerminationsCfg,
    make_sweep_policy_robot_cfg,
)
from sweep_rl.shelf_reach.mdp.rewards import tcp_orientation_command_error

from . import mdp

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0"
TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1"
PLAY_TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v1"
TASK_ID_V2 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2"
PLAY_TASK_ID_V2 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v2"
TASK_ID_V3 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v3"
PLAY_TASK_ID_V3 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v3"
TASK_ID_V4 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v4"
PLAY_TASK_ID_V4 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v4"

PRE_REACH_X_OFFSET = 0.0
PUSH_GOAL_OFFSET = (0.0, 0.18, 0.0)
PUSH_REWARD_WEIGHT = 6.0
PUSH_REWARD_WEIGHT_V1 = 2.0 * PUSH_REWARD_WEIGHT
SWEEP_GATE_DISTANCE = 0.08
WRIST_BACK_OFFSET = -0.14

# V1 stabilizes the object without changing its height or nominal contact z.
CUBE_WIDTH_V1 = 0.11
CUBE_DEPTH_V1 = 0.11
CUBE_COM_OFFSET_V1 = (0.0, 0.0, -0.05)
CUBE_DIAGONAL_INERTIA_V1 = (0.008, 0.008, 0.003)
CUBE_MAX_TILT_RADIANS_V1 = math.radians(10.0)
VERTICAL_WRIST_3_POSITION_V1 = math.pi / 2.0
ARM_TARGET_MAX_DELTA_V1 = 0.05
WRIST_TARGET_MAX_DELTA_V1 = 0.07

# V2 returns to the original footprint and further slows robot/object motion.
CUBE_WIDTH_V2 = CUBE_WIDTH
CUBE_DEPTH_V2 = CUBE_DEPTH
MAX_EPISODE_STEPS_V2 = 360
ARM_TARGET_MAX_DELTA_V2 = 0.03
WRIST_TARGET_MAX_DELTA_V2 = 0.04
ARM_VELOCITY_LIMIT_V2 = 1.5
WRIST_VELOCITY_LIMIT_V2 = 2.0
PUSH_VELOCITY_REWARD_MIN_V2 = 0.03
PUSH_VELOCITY_PENALTY_MIN_V2 = 0.06
PUSH_BEHIND_WIDTH_SCALE_V4 = 1.0
SWEEP_GATE_DISTANCE_V4 = 0.04
PUSH_VELOCITY_REWARD_MIN_V4 = 0.05
PUSH_VELOCITY_PENALTY_MIN_V4 = 0.10
SHELF_COLLISION_WEIGHT_V2 = 2.0 * SHELF_COLLISION_WEIGHT
SHELF_FLOOR_SURFACE_HEIGHTS_V2 = (0.70, 1.05, 1.50)
SELF_COLLISION_SOFT_FORCE_THRESHOLD_V2 = 2.0
SELF_COLLISION_PERSISTENT_FORCE_THRESHOLD_V2 = 5.0
SELF_COLLISION_HARD_FORCE_THRESHOLD_V2 = 20.0
SELF_COLLISION_PERSISTENCE_STEPS_V2 = 2
SELF_COLLISION_FORCE_PENALTY_WEIGHT_V2 = -1.0
SELF_COLLISION_TERMINAL_BASE_COST_V2 = 10.0
SELF_COLLISION_TERMINAL_EARLY_COST_V2 = 30.0

SELF_COLLISION_SENSOR_NAMES_V2 = (
    "base_self_contact",
    "shoulder_self_contact",
    "upper_arm_self_contact",
    "forearm_self_contact",
    "wrist_1_self_contact",
    "wrist_2_self_contact",
    "wrist_3_self_contact",
    "gripper_base_self_contact",
    "left_outer_knuckle_self_contact",
    "left_outer_finger_self_contact",
    "left_inner_finger_self_contact",
    "left_inner_knuckle_self_contact",
    "right_outer_knuckle_self_contact",
    "right_outer_finger_self_contact",
    "right_inner_finger_self_contact",
    "right_inner_knuckle_self_contact",
)

# Directly connected arm links and internal gripper linkage pairs are not failures.
# Every other arm--arm or gripper--arm pair remains a self-collision candidate.
SELF_COLLISION_EXCLUDED_PAIRS_V2 = (
    ("base_self_contact", "shoulder_self_contact"),
    ("shoulder_self_contact", "upper_arm_self_contact"),
    ("upper_arm_self_contact", "forearm_self_contact"),
    ("forearm_self_contact", "wrist_1_self_contact"),
    ("wrist_1_self_contact", "wrist_2_self_contact"),
    ("wrist_2_self_contact", "wrist_3_self_contact"),
    ("wrist_3_self_contact", "gripper_base_self_contact"),
    ("gripper_base_self_contact", "left_outer_knuckle_self_contact"),
    ("gripper_base_self_contact", "left_inner_knuckle_self_contact"),
    ("gripper_base_self_contact", "right_outer_knuckle_self_contact"),
    ("gripper_base_self_contact", "right_inner_knuckle_self_contact"),
    ("left_outer_knuckle_self_contact", "left_outer_finger_self_contact"),
    ("left_inner_knuckle_self_contact", "left_inner_finger_self_contact"),
    ("left_outer_finger_self_contact", "left_inner_finger_self_contact"),
    ("right_outer_knuckle_self_contact", "right_outer_finger_self_contact"),
    ("right_inner_knuckle_self_contact", "right_inner_finger_self_contact"),
    ("right_outer_finger_self_contact", "right_inner_finger_self_contact"),
)

if len(SELF_COLLISION_SENSOR_NAMES_V2) != len(ROBOT_CONTACT_BODY_PATHS):
    raise ValueError("Every v2 robot rigid body must have one self-contact sensor.")

if not 0.0 < SWEEP_GATE_DISTANCE < CUBE_WIDTH * BEHIND_WIDTH_SCALE:
    raise ValueError(
        "Sweep gate distance must be positive and smaller than the pre-reach Y offset."
    )
if not SWEEP_GATE_DISTANCE < CUBE_WIDTH_V1 * BEHIND_WIDTH_SCALE:
    raise ValueError(
        "Sweep gate distance must be smaller than the v1 pre-reach Y offset."
    )


@configclass
class ShelfCubeSweepSceneCfg(ShelfCubePreReachSceneCfg):
    """PreReach scene extended only with the wrist reference frame."""

    wrist_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="wrist",
                offset=OffsetCfg(pos=(WRIST_BACK_OFFSET, 0.0, 0.0)),
            )
        ],
    )


def make_vertical_sweep_policy_robot_cfg():
    """Return the shared robot with a vertical v1 wrist reset posture."""
    robot_cfg = make_sweep_policy_robot_cfg()
    robot_cfg.init_state.joint_pos["wrist_3_joint"] = VERTICAL_WRIST_3_POSITION_V1
    return robot_cfg


@configclass
class ShelfCubeSweepSceneCfgV1(ShelfCubeSweepSceneCfg):
    """V1 scene with a wider, explicitly bottom-heavy Cube."""

    robot = make_vertical_sweep_policy_robot_cfg()
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=mdp.BottomHeavyCuboidCfg(
            size=(CUBE_WIDTH_V1, CUBE_DEPTH_V1, CUBE_HEIGHT),
            center_of_mass=CUBE_COM_OFFSET_V1,
            diagonal_inertia=CUBE_DIAGONAL_INERTIA_V1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.70, -0.10, CUBE_CENTER_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_v2_sweep_policy_robot_cfg():
    """Return the vertical robot with lower v2 actuator velocity limits."""
    robot_cfg = make_vertical_sweep_policy_robot_cfg()
    robot_cfg.actuators["arm"].velocity_limit_sim = {
        "shoulder_pan_joint": ARM_VELOCITY_LIMIT_V2,
        "shoulder_lift_joint": ARM_VELOCITY_LIMIT_V2,
        "elbow_joint": ARM_VELOCITY_LIMIT_V2,
        "wrist_1_joint": WRIST_VELOCITY_LIMIT_V2,
        "wrist_2_joint": WRIST_VELOCITY_LIMIT_V2,
        "wrist_3_joint": WRIST_VELOCITY_LIMIT_V2,
    }
    return robot_cfg


def make_v2_self_collision_robot_cfg():
    """Retained opt-in robot cfg for the currently disabled self-collision check."""
    robot_cfg = make_v2_sweep_policy_robot_cfg()
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True
    )
    return robot_cfg


def make_v2_robot_self_contact_sensor(body_path: str) -> ContactSensorCfg:
    """Create a single-body sensor filtered against every robot rigid body."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_path}",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_contact_points=False,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{filter_body_path}"
            for filter_body_path in ROBOT_CONTACT_BODY_PATHS
        ],
        debug_vis=False,
    )


@configclass
class ShelfCubeSweepSceneCfgV2(ShelfCubeSweepSceneCfgV1):
    """V2 scene with the original footprint and retained low COM."""

    robot = make_v2_sweep_policy_robot_cfg()
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=mdp.BottomHeavyCuboidCfg(
            size=(CUBE_WIDTH_V2, CUBE_DEPTH_V2, CUBE_HEIGHT),
            center_of_mass=CUBE_COM_OFFSET_V1,
            diagonal_inertia=CUBE_DIAGONAL_INERTIA_V1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.70, -0.10, CUBE_CENTER_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.shelf.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )


@configclass
class ShelfCubeSweepSelfCollisionSceneCfgV2(ShelfCubeSweepSceneCfgV2):
    """Retained opt-in self-contact scene; not wired into the registered v2 task."""

    robot = make_v2_self_collision_robot_cfg()
    base_self_contact = make_v2_robot_self_contact_sensor("base_link")
    shoulder_self_contact = make_v2_robot_self_contact_sensor("shoulder_link")
    upper_arm_self_contact = make_v2_robot_self_contact_sensor("upper_arm_link")
    forearm_self_contact = make_v2_robot_self_contact_sensor("forearm_link")
    wrist_1_self_contact = make_v2_robot_self_contact_sensor("wrist_1_link")
    wrist_2_self_contact = make_v2_robot_self_contact_sensor("wrist_2_link")
    wrist_3_self_contact = make_v2_robot_self_contact_sensor("wrist_3_link")
    gripper_base_self_contact = make_v2_robot_self_contact_sensor("robotiq_base_link")
    left_outer_knuckle_self_contact = make_v2_robot_self_contact_sensor(
        "left_outer_knuckle"
    )
    left_outer_finger_self_contact = make_v2_robot_self_contact_sensor(
        "left_outer_finger"
    )
    left_inner_finger_self_contact = make_v2_robot_self_contact_sensor(
        "left_inner_finger"
    )
    left_inner_knuckle_self_contact = make_v2_robot_self_contact_sensor(
        "left_inner_knuckle"
    )
    right_outer_knuckle_self_contact = make_v2_robot_self_contact_sensor(
        "right_outer_knuckle"
    )
    right_outer_finger_self_contact = make_v2_robot_self_contact_sensor(
        "right_outer_finger"
    )
    right_inner_finger_self_contact = make_v2_robot_self_contact_sensor(
        "right_inner_finger"
    )
    right_inner_knuckle_self_contact = make_v2_robot_self_contact_sensor(
        "right_inner_knuckle"
    )


@configclass
class CubeSweepActionsCfgV1:
    """Full-range absolute arm targets with a control-step slew limit."""

    arm_action = mdp.RateLimitedJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=1.0,
        rescale_to_limits=True,
        preserve_order=True,
        max_delta={
            "shoulder_.*": ARM_TARGET_MAX_DELTA_V1,
            "elbow_joint": ARM_TARGET_MAX_DELTA_V1,
            "wrist_.*": WRIST_TARGET_MAX_DELTA_V1,
        },
    )


@configclass
class CubeSweepActionsCfgV2:
    """V2 full-range targets with slower per-step target motion."""

    arm_action = mdp.RateLimitedJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=1.0,
        rescale_to_limits=True,
        preserve_order=True,
        max_delta={
            "shoulder_.*": ARM_TARGET_MAX_DELTA_V2,
            "elbow_joint": ARM_TARGET_MAX_DELTA_V2,
            "wrist_.*": WRIST_TARGET_MAX_DELTA_V2,
        },
    )


@configclass
class CubeSweepActionsCfgV4:
    """Original Sweep-Policy default-relative arm and binary gripper actions."""

    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GRIPPER_JOINT_NAMES),
        open_command_expr={
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
        },
        close_command_expr={
            "finger_joint": 0.5,
            "right_outer_knuckle_joint": 0.5,
            "left_outer_finger_joint": 0.0,
            "left_inner_finger_knuckle_joint": -0.5,
            "left_inner_finger_joint": -0.5,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.5,
            "right_inner_finger_knuckle_joint": -0.5,
        },
    )


ARM_RESET_CFG_V1 = SceneEntityCfg(
    "robot",
    joint_names=list(ARM_JOINT_NAMES[:3]),
    preserve_order=True,
)


@configclass
class CubeSweepEventsCfgV1(EventCfg):
    """Keep all wrist joints at their vertical defaults during reset."""

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": ARM_RESET_CFG_V1,
        },
    )


@configclass
class CubeSweepEventsCfgV2(CubeSweepEventsCfgV1):
    """Use a narrow collision-free shoulder/elbow reset neighborhood in v2."""

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": ARM_RESET_CFG_V1,
        },
    )


@configclass
class CubeSweepObservationsCfgV1(CubePreReachObservationsCfg):
    """Retain the original policy inputs with the wider v1 Cube constant."""

    @configclass
    class PolicyCfg(CubePreReachObservationsCfg.PolicyCfg):
        cube_width = ObsTerm(
            func=cube_width_observation,
            params={"width": CUBE_WIDTH_V1},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubeSweepObservationsCfgV2(CubePreReachObservationsCfg):
    """Original 29 policy inputs with the rolled-back Cube width."""

    @configclass
    class PolicyCfg(CubePreReachObservationsCfg.PolicyCfg):
        cube_width = ObsTerm(
            func=cube_width_observation,
            params={"width": CUBE_WIDTH_V2},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubeSweepObservationsCfgV4(CubeSweepObservationsCfgV2):
    """V2 policy inputs extended with the current 7D TCP pose."""

    @configclass
    class PolicyCfg(CubeSweepObservationsCfgV2.PolicyCfg):
        current_ee_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={
                "robot_cfg": ROBOT_CFG,
                "frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubeSweepCommandsCfg(CubePreReachCommandsCfg):
    """PreReach moving command extended with a fixed episode Sweep goal."""

    ee_pose = mdp.CubeSweepPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        object_name="target_object",
        cube_width=CUBE_WIDTH,
        behind_width_scale=BEHIND_WIDTH_SCALE,
        z_offset=PRE_REACH_Z_OFFSET,
        target_roll=TARGET_ROLL[0],
        target_pitch=TARGET_PITCH[0],
        target_yaw=TARGET_YAW[0],
        goal_offset=PUSH_GOAL_OFFSET,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.CubeSweepPoseCommandCfg.Ranges(
            pos_x=TARGET_POS_X,
            pos_y=TARGET_POS_Y,
            pos_z=TARGET_POS_Z,
            roll=TARGET_ROLL,
            pitch=TARGET_PITCH,
            yaw=TARGET_YAW,
        ),
    )


@configclass
class CubeSweepCommandsCfgV1(CubePreReachCommandsCfg):
    """V1 Sweep command using the wider Cube footprint."""

    ee_pose = mdp.CubeSweepPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        object_name="target_object",
        cube_width=CUBE_WIDTH_V1,
        behind_width_scale=BEHIND_WIDTH_SCALE,
        z_offset=PRE_REACH_Z_OFFSET,
        target_roll=TARGET_ROLL[0],
        target_pitch=TARGET_PITCH[0],
        target_yaw=TARGET_YAW[0],
        goal_offset=PUSH_GOAL_OFFSET,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.CubeSweepPoseCommandCfg.Ranges(
            pos_x=TARGET_POS_X,
            pos_y=TARGET_POS_Y,
            pos_z=TARGET_POS_Z,
            roll=TARGET_ROLL,
            pitch=TARGET_PITCH,
            yaw=TARGET_YAW,
        ),
    )


@configclass
class CubeSweepCommandsCfgV2(CubePreReachCommandsCfg):
    """V2 Sweep command using the original Cube footprint."""

    ee_pose = mdp.CubeSweepPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        object_name="target_object",
        cube_width=CUBE_WIDTH_V2,
        behind_width_scale=BEHIND_WIDTH_SCALE,
        z_offset=PRE_REACH_Z_OFFSET,
        target_roll=TARGET_ROLL[0],
        target_pitch=TARGET_PITCH[0],
        target_yaw=TARGET_YAW[0],
        goal_offset=PUSH_GOAL_OFFSET,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.CubeSweepPoseCommandCfg.Ranges(
            pos_x=TARGET_POS_X,
            pos_y=TARGET_POS_Y,
            pos_z=TARGET_POS_Z,
            roll=TARGET_ROLL,
            pitch=TARGET_PITCH,
            yaw=TARGET_YAW,
        ),
    )


@configclass
class CubeSweepRewardsCfg(CubePreReachRewardsCfg):
    """Inherit Reach/orientation/collision terms and add Sweep shaping."""

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": BEHIND_WIDTH_SCALE,
            "z_offset": PRE_REACH_Z_OFFSET,
            "gate_distance": SWEEP_GATE_DISTANCE,
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )


@configclass
class CubeSweepRewardsCfgV1(CubePreReachRewardsCfgV1):
    """Keep all PreReach-v1 weights and add double-weight Sweep shaping."""

    end_effector_orientation_tracking = RewTerm(
        func=tcp_orientation_command_error,
        weight=-1.5,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT_V1,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH_V1,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": BEHIND_WIDTH_SCALE,
            "z_offset": PRE_REACH_Z_OFFSET,
            "gate_distance": SWEEP_GATE_DISTANCE,
            "max_tilt_radians": CUBE_MAX_TILT_RADIANS_V1,
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )


@configclass
class CubeSweepRewardsCfgV2(CubeSweepRewardsCfgV1):
    """V1 safety rewards with slower per-step pushing velocity shaping."""

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT_V1,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH_V2,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": BEHIND_WIDTH_SCALE,
            "z_offset": PRE_REACH_Z_OFFSET,
            "gate_distance": SWEEP_GATE_DISTANCE,
            "max_tilt_radians": CUBE_MAX_TILT_RADIANS_V1,
            "velocity_reward_min": PUSH_VELOCITY_REWARD_MIN_V2,
            "velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN_V2,
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )

    shelf_collision = RewTerm(
        func=cube_pre_reach_mdp.shelf_collision,
        weight=SHELF_COLLISION_WEIGHT_V2,
        params={
            "sensor_name": "shelf_floor_contact",
            "force_threshold": SHELF_CONTACT_FORCE_THRESHOLD,
            "surface_heights": SHELF_FLOOR_SURFACE_HEIGHTS_V2,
            "surface_tolerance": SHELF_FLOOR_SURFACE_TOLERANCE,
            "x_bounds": SHELF_FLOOR_X_BOUNDS,
            "y_bounds": SHELF_FLOOR_Y_BOUNDS,
            "shelf_cfg": SHELF_CFG,
        },
    )


@configclass
class CubeSweepRewardsCfgV3(CubeSweepRewardsCfgV2):
    """V2 rewards using Cube actor-origin planar speed for push shaping."""

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT_V1,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH_V2,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": BEHIND_WIDTH_SCALE,
            "z_offset": PRE_REACH_Z_OFFSET,
            "gate_distance": SWEEP_GATE_DISTANCE,
            "max_tilt_radians": CUBE_MAX_TILT_RADIANS_V1,
            "velocity_reward_min": PUSH_VELOCITY_REWARD_MIN_V2,
            "velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN_V2,
            "velocity_measurement": "root_link_planar",
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )


@configclass
class CubeSweepRewardsCfgV4(CubeSweepRewardsCfgV3):
    """Original directional speed band and contact geometry without tilt scaling."""

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT_V1,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH_V2,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": PUSH_BEHIND_WIDTH_SCALE_V4,
            "z_offset": PRE_REACH_Z_OFFSET,
            "gate_distance": SWEEP_GATE_DISTANCE_V4,
            "velocity_reward_min": PUSH_VELOCITY_REWARD_MIN_V4,
            "velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN_V4,
            "velocity_measurement": "com_y",
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )


@configclass
class CubeSweepSelfCollisionRewardsCfgV2(CubeSweepRewardsCfgV2):
    """Retained opt-in self-collision rewards; not wired into registered v2."""

    self_collision_force = RewTerm(
        func=mdp.self_collision_force_penalty,
        weight=SELF_COLLISION_FORCE_PENALTY_WEIGHT_V2,
        params={
            "soft_force_threshold": SELF_COLLISION_SOFT_FORCE_THRESHOLD_V2,
            "hard_force_threshold": SELF_COLLISION_HARD_FORCE_THRESHOLD_V2,
        },
    )

    self_collision_terminal = RewTerm(
        func=mdp.self_collision_terminal_penalty,
        weight=-1.0,
        params={
            "termination_term_name": "self_collision",
            "base_cost": SELF_COLLISION_TERMINAL_BASE_COST_V2,
            "early_termination_cost": SELF_COLLISION_TERMINAL_EARLY_COST_V2,
        },
    )


@configclass
class CubeSweepTerminationsCfgV2(TerminationsCfg):
    """Retained opt-in self-collision termination; not wired into registered v2."""

    self_collision = DoneTerm(
        func=mdp.SelfCollisionTermination,
        time_out=False,
        params={
            "sensor_names": SELF_COLLISION_SENSOR_NAMES_V2,
            "excluded_pairs": SELF_COLLISION_EXCLUDED_PAIRS_V2,
            "command_name": "ee_pose",
            "persistent_force_threshold": (
                SELF_COLLISION_PERSISTENT_FORCE_THRESHOLD_V2
            ),
            "hard_force_threshold": SELF_COLLISION_HARD_FORCE_THRESHOLD_V2,
            "persistence_steps": SELF_COLLISION_PERSISTENCE_STEPS_V2,
        },
    )


@configclass
class UR5eGripperShelfCubeSweepEnvCfg(UR5eGripperShelfCubePreReachEnvCfg):
    """PreReach task extended with implicit +Y Cube pushing."""

    scene: ShelfCubeSweepSceneCfg = ShelfCubeSweepSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    commands: CubeSweepCommandsCfg = CubeSweepCommandsCfg()
    rewards: CubeSweepRewardsCfg = CubeSweepRewardsCfg()


@configclass
class UR5eGripperShelfCubeSweepEnvCfg_PLAY(UR5eGripperShelfCubeSweepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV1(UR5eGripperShelfCubePreReachEnvCfgV1):
    """CubePreReach-v1 extended with the moving-offset Sweep reward."""

    scene: ShelfCubeSweepSceneCfgV1 = ShelfCubeSweepSceneCfgV1(
        num_envs=4096, env_spacing=2.5
    )
    actions: CubeSweepActionsCfgV1 = CubeSweepActionsCfgV1()
    commands: CubeSweepCommandsCfgV1 = CubeSweepCommandsCfgV1()
    observations: CubeSweepObservationsCfgV1 = CubeSweepObservationsCfgV1()
    rewards: CubeSweepRewardsCfgV1 = CubeSweepRewardsCfgV1()
    events: CubeSweepEventsCfgV1 = CubeSweepEventsCfgV1()
    curriculum = None


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV2(UR5eGripperShelfCubeSweepEnvCfgV1):
    """Slower Sweep task with the original Cube footprint."""

    scene: ShelfCubeSweepSceneCfgV2 = ShelfCubeSweepSceneCfgV2(
        num_envs=4096, env_spacing=2.5
    )
    actions: CubeSweepActionsCfgV2 = CubeSweepActionsCfgV2()
    commands: CubeSweepCommandsCfgV2 = CubeSweepCommandsCfgV2()
    observations: CubeSweepObservationsCfgV2 = CubeSweepObservationsCfgV2()
    rewards: CubeSweepRewardsCfgV2 = CubeSweepRewardsCfgV2()
    # Self-collision scene/rewards/termination remain implemented above, but are
    # intentionally not connected while reproducing the pre-stabilization v2.
    events: CubeSweepEventsCfgV1 = CubeSweepEventsCfgV1()

    def __post_init__(self):
        super().__post_init__()
        episode_duration = MAX_EPISODE_STEPS_V2 * self.decimation * self.sim.dt
        self.episode_length_s = math.nextafter(episode_duration, 0.0)


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV3(UR5eGripperShelfCubeSweepEnvCfgV2):
    """V2 task with actor-origin planar push speed and fixed-rate PPO."""

    rewards: CubeSweepRewardsCfgV3 = CubeSweepRewardsCfgV3()


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV4(UR5eGripperShelfCubeSweepEnvCfgV3):
    """V3 scene with original actions, current TCP observation, and corrected push shaping."""

    actions: CubeSweepActionsCfgV4 = CubeSweepActionsCfgV4()
    observations: CubeSweepObservationsCfgV4 = CubeSweepObservationsCfgV4()
    rewards: CubeSweepRewardsCfgV4 = CubeSweepRewardsCfgV4()


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV1_PLAY(UR5eGripperShelfCubeSweepEnvCfgV1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV2_PLAY(UR5eGripperShelfCubeSweepEnvCfgV2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV3_PLAY(UR5eGripperShelfCubeSweepEnvCfgV3):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True


@configclass
class UR5eGripperShelfCubeSweepEnvCfgV4_PLAY(UR5eGripperShelfCubeSweepEnvCfgV4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
