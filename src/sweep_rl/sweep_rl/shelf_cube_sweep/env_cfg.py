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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from sweep_rl.shelf_cube_pre_reach.env_cfg import (
    BEHIND_WIDTH_SCALE,
    CUBE_CENTER_HEIGHT,
    CUBE_HEIGHT,
    CUBE_MASS,
    CUBE_WIDTH,
    CubePreReachObservationsCfg,
    EE_FRAME_CFG,
    GRIPPER_BASE_BODY_PATH,
    PRE_REACH_Z_OFFSET,
    ROBOT_CFG,
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
from sweep_rl.shelf_reach.env_cfg import (
    ARM_JOINT_NAMES,
    EventCfg,
    ROBOT_BASE_BODY_PATH,
    make_sweep_policy_robot_cfg,
)
from sweep_rl.shelf_reach.mdp.rewards import tcp_orientation_command_error

from . import mdp

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0"
TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1"
PLAY_TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v1"

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
    robot_cfg.init_state.joint_pos["wrist_3_joint"] = (
        VERTICAL_WRIST_3_POSITION_V1
    )
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
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.70, -0.10, CUBE_CENTER_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
class UR5eGripperShelfCubeSweepEnvCfgV1(
    UR5eGripperShelfCubePreReachEnvCfgV1
):
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
class UR5eGripperShelfCubeSweepEnvCfgV1_PLAY(
    UR5eGripperShelfCubeSweepEnvCfgV1
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
