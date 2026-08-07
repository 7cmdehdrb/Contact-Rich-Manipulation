"""Cube-relative pre-reaching task inherited from UR5e shelf reach."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from sweep_rl.shelf_reach.env_cfg import (
    ARM_CFG,
    GRIPPER_BASE_BODY_PATH,
    TARGET_PITCH,
    TARGET_POS_X,
    TARGET_POS_Y,
    TARGET_POS_Z,
    TARGET_ROLL,
    TARGET_YAW,
    RewardsCfg,
    ShelfReachSceneCfg,
    UR5eGripperShelfReachEnvCfg,
)

EE_FRAME_CFG = SceneEntityCfg("ee_frame")

from . import mdp

from sweep_rl.shelf_reach.mdp.rewards import (
    tcp_position_command_error,
    _desired_pose_w,
    tcp_orientation_command_error,
)

TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0"
PLAY_TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0"
TASK_ID_V1 = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1"
PLAY_TASK_ID_V1 = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v1"

CUBE_WIDTH = 0.06
CUBE_DEPTH = 0.06
CUBE_HEIGHT = 0.15
CUBE_MASS = 0.50
SHELF_SURFACE_HEIGHT = 1.05
CUBE_CENTER_HEIGHT = SHELF_SURFACE_HEIGHT + 0.5 * CUBE_HEIGHT
CUBE_INITIAL_POSITION = (-0.70, -0.10, CUBE_CENTER_HEIGHT)
BEHIND_WIDTH_SCALE = 1.2
PRE_REACH_Z_OFFSET = 0.03
SHELF_FLOOR_BODY_PATH = "rack"
SHELF_FLOOR_X_BOUNDS = (-0.20, 0.20)
SHELF_FLOOR_Y_BOUNDS = (-0.50, 0.50)
SHELF_FLOOR_SURFACE_HEIGHT = 1.05
SHELF_FLOOR_SURFACE_TOLERANCE = 0.02
SHELF_CONTACT_FORCE_THRESHOLD = 1.0
SHELF_COLLISION_WEIGHT = -0.05
PRE_REACH_EXP_REWARD_WEIGHT = 3.0

# Every rigid body in the combined UR5e + Robotiq asset.  The shelf-side
# sensor uses one-to-many filtering so Cube--shelf and shelf self contacts are
# never included in the collision reward.
ROBOT_CONTACT_BODY_PATHS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "robotiq_base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_knuckle",
)

ROBOT_CFG = SceneEntityCfg("robot")
TARGET_OBJECT_CFG = SceneEntityCfg("target_object")
SHELF_CFG = SceneEntityCfg("shelf")


@configclass
class ShelfCubePreReachSceneCfg(ShelfReachSceneCfg):
    """Parent shelf scene with one physical Cube added."""

    shelf_floor_contact = ContactSensorCfg(
        # The shelf USD has one rigid body (rack) containing all its collision
        # shapes.  The reward further masks contact points to Cube_02, the
        # horizontal board whose top surface is at z=1.05 in shelf coordinates.
        prim_path=f"{{ENV_REGEX_NS}}/Shelf/{SHELF_FLOOR_BODY_PATH}",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{body_path}"
            for body_path in ROBOT_CONTACT_BODY_PATHS
        ],
        debug_vis=False,
    )

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_WIDTH, CUBE_DEPTH, CUBE_HEIGHT),
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
            pos=CUBE_INITIAL_POSITION,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.shelf.spawn.activate_contact_sensors = True


@configclass
class CubePreReachCommandsCfg:
    ee_pose = mdp.CubePreReachPoseCommandCfg(
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
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        # Required by the parent config but intentionally unused by the
        # Cube-relative command implementation.
        ranges=mdp.CubePreReachPoseCommandCfg.Ranges(
            pos_x=TARGET_POS_X,
            pos_y=TARGET_POS_Y,
            pos_z=TARGET_POS_Z,
            roll=TARGET_ROLL,
            pitch=TARGET_PITCH,
            yaw=TARGET_YAW,
        ),
    )


@configclass
class CubePreReachObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        pose_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        target_object_position = ObsTerm(
            func=mdp.target_object_position_b,
            params={
                "robot_cfg": ROBOT_CFG,
                "object_cfg": TARGET_OBJECT_CFG,
            },
        )
        cube_width = ObsTerm(
            func=mdp.cube_width,
            params={"width": CUBE_WIDTH},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubePreReachRewardsCfg(RewardsCfg):
    """Parent Reach rewards plus a 20%-scale robot/floor contact penalty."""

    shelf_collision = RewTerm(
        func=mdp.shelf_collision,
        weight=SHELF_COLLISION_WEIGHT,
        params={
            "sensor_name": "shelf_floor_contact",
            "force_threshold": SHELF_CONTACT_FORCE_THRESHOLD,
            "surface_height": SHELF_FLOOR_SURFACE_HEIGHT,
            "surface_tolerance": SHELF_FLOOR_SURFACE_TOLERANCE,
            "x_bounds": SHELF_FLOOR_X_BOUNDS,
            "y_bounds": SHELF_FLOOR_Y_BOUNDS,
            "shelf_cfg": SHELF_CFG,
        },
    )


@configclass
class CubePreReachRewardsCfgV1(CubePreReachRewardsCfg):
    """Replace both parent position terms with one exponential Reach reward."""

    # Tracking 에러 기준 변경 -> Max 1.0, 태호 기준 3.0
    end_effector_position_tracking = RewTerm(
        func=mdp.tcp_position_command_reward_exp,
        weight=3.0,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    end_effector_position_tracking_fine_grained = None

    # Weight 변경 -> Max 1.0 -> 3.14
    end_effector_orientation_tracking = RewTerm(
        func=mdp.tcp_orientation_command_error,
        weight=-0.7,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )

    # Weight 변경
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": ARM_CFG},
    )


@configclass
class UR5eGripperShelfCubePreReachEnvCfg(UR5eGripperShelfReachEnvCfg):
    """Inherited shelf reach with a physical Cube-relative pre-reach goal."""

    scene: ShelfCubePreReachSceneCfg = ShelfCubePreReachSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    commands: CubePreReachCommandsCfg = CubePreReachCommandsCfg()
    observations: CubePreReachObservationsCfg = CubePreReachObservationsCfg()
    rewards: CubePreReachRewardsCfg = CubePreReachRewardsCfg()


@configclass
class UR5eGripperShelfCubePreReachEnvCfg_PLAY(UR5eGripperShelfCubePreReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True


@configclass
class UR5eGripperShelfCubePreReachEnvCfgV1(UR5eGripperShelfCubePreReachEnvCfg):
    """PreReach v1 using ``3 * exp(-10 * EE offset distance)``."""

    rewards: CubePreReachRewardsCfgV1 = CubePreReachRewardsCfgV1()


@configclass
class UR5eGripperShelfCubePreReachEnvCfgV1_PLAY(UR5eGripperShelfCubePreReachEnvCfgV1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
