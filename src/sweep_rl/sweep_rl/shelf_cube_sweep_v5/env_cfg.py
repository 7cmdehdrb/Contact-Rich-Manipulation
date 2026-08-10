"""Standalone UR5e/Robotiq shelf Cube Sweep v5 environment."""

from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v5"

DEFAULT_ROBOT_USD_PATH = "./asset/Shelf_USD/Robots/UR5e/Collected_UR5e_v4/UR5e_v4.usd"
DEFAULT_SHELF_USD_PATH = (
    "omniverse://192.168.0.13/Library/Shelf/Arena/"
    "Collected_speedrack_shape/speedrack_shape.usd"
)

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT_NAMES = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
)
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

ROBOT_BASE_BODY_PATH = "base_link"
GRIPPER_BASE_BODY_PATH = "robotiq_base_link"
SHELF_FLOOR_BODY_PATH = "rack"
TCP_FORWARD_OFFSET = 0.13
WRIST_BACK_OFFSET = -0.14

CUBE_WIDTH = 0.08
CUBE_DEPTH = 0.08
CUBE_HEIGHT = 0.20
CUBE_MASS = 1.5
ACTIVE_SHELF_SURFACE_HEIGHT = 1.05
CUBE_CENTER_HEIGHT = ACTIVE_SHELF_SURFACE_HEIGHT + 0.5 * CUBE_HEIGHT
CUBE_INITIAL_POSITION = (-0.70, -0.10, CUBE_CENTER_HEIGHT)
CUBE_COM_OFFSET = (0.0, 0.0, -0.08)
CUBE_DIAGONAL_INERTIA = (0.008, 0.008, 0.003)
CUBE_DROP_MIN_CENTER_HEIGHT = ACTIVE_SHELF_SURFACE_HEIGHT - 0.01
CUBE_MAX_TILT_RADIANS = 0.9
CUBE_MAX_LINEAR_SPEED = 0.3
ARM_MAX_JOINT_SPEED = 1.0

PUSH_GOAL_OFFSET = (0.0, 0.18, 0.0)
CONTACT_POINT_BEHIND_WIDTH_SCALE = 0.5
CONTACT_POINT_X_OFFSET = 0.0
CONTACT_POINT_Z_OFFSET = 0.0
SWEEP_GATE_DISTANCE = 0.04
PUSH_VELOCITY_REWARD_MIN = 0.05
PUSH_VELOCITY_PENALTY_MIN = 0.10
REACH_REWARD_WEIGHT = 3.0
PUSH_REWARD_WEIGHT = 2.0 * REACH_REWARD_WEIGHT
ORIENTATION_REWARD_WEIGHT = 2.0
HOME_SUCCESS_DISTANCE = 0.03
HOME_GATE_SHARPNESS = 100.0
HOME_REWARD_WEIGHT = 9.0

MAX_EPISODE_STEPS = 500

SHELF_FLOOR_X_BOUNDS = (-0.20, 0.20)
SHELF_FLOOR_Y_BOUNDS = (-0.50, 0.50)
SHELF_FLOOR_SURFACE_HEIGHTS = (0.70, ACTIVE_SHELF_SURFACE_HEIGHT, 1.50)
SHELF_FLOOR_SURFACE_TOLERANCE = 0.02
SHELF_CONTACT_FORCE_THRESHOLD = 1.0
SHELF_COLLISION_WEIGHT = -10.0

TARGET_POS_X = (0.55, 0.75)
TARGET_POS_Y = (-0.20, 0.20)
TARGET_POS_Z = (0.30, 0.50)
TARGET_ROLL = (math.pi / 2.0, math.pi / 2.0)
TARGET_PITCH = (0.0, 0.0)
TARGET_YAW = (0.0, 0.0)

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
HOME_ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES[:5]), preserve_order=True
)
ARM_RESET_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES[:3]), preserve_order=True
)
ROBOT_CFG = SceneEntityCfg("robot")
TARGET_OBJECT_CFG = SceneEntityCfg("target_object")
SHELF_CFG = SceneEntityCfg("shelf")
EE_FRAME_CFG = SceneEntityCfg("ee_frame")
WRIST_FRAME_CFG = SceneEntityCfg("wrist_frame")


def make_robot_cfg() -> ArticulationCfg:
    """Create the final v5 robot directly, without another task config."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEP_POLICY_ROBOT_USD_PATH", DEFAULT_ROBOT_USD_PATH
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.79505),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.2,
                "elbow_joint": 2.2,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": math.pi / 2.0,
                ".*(finger|knuckle).*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit_sim={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                velocity_limit_sim={
                    "shoulder_pan_joint": 1.5,
                    "shoulder_lift_joint": 1.5,
                    "elbow_joint": 1.5,
                    "wrist_1_joint": 2.0,
                    "wrist_2_joint": 2.0,
                    "wrist_3_joint": 2.0,
                },
                stiffness={
                    "shoulder_pan_joint": 3328.7,
                    "shoulder_lift_joint": 4989.4,
                    "elbow_joint": 2394.2,
                    "wrist_1_joint": 3500.2,
                    "wrist_2_joint": 3174.8,
                    "wrist_3_joint": 3032.4,
                },
                damping={
                    "shoulder_pan_joint": 368.8,
                    "shoulder_lift_joint": 561.2,
                    "elbow_joint": 274.1,
                    "wrist_1_joint": 396.2,
                    "wrist_2_joint": 363.1,
                    "wrist_3_joint": 345.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_JOINT_NAMES),
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=2000.0,
                damping=1000.0,
            ),
        },
    )


@configclass
class ShelfCubeSweepV5SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("SWEEP_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )
    robot = make_robot_cfg()
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=mdp.BottomHeavyCuboidCfg(
            size=(CUBE_WIDTH, CUBE_DEPTH, CUBE_HEIGHT),
            center_of_mass=CUBE_COM_OFFSET,
            diagonal_inertia=CUBE_DIAGONAL_INERTIA,
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
            pos=CUBE_INITIAL_POSITION, rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    ee_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="tcp",
                offset=OffsetCfg(pos=(TCP_FORWARD_OFFSET, 0.0, 0.0)),
            )
        ],
    )
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
    shelf_floor_contact = ContactSensorCfg(
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
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    ee_pose = mdp.CubeSweepPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        object_name="target_object",
        cube_width=CUBE_WIDTH,
        behind_width_scale=CONTACT_POINT_BEHIND_WIDTH_SCALE,
        x_offset=CONTACT_POINT_X_OFFSET,
        z_offset=CONTACT_POINT_Z_OFFSET,
        push_gate_distance=SWEEP_GATE_DISTANCE,
        wrist_frame_name="wrist_frame",
        wrist_frame_index=0,
        target_roll=TARGET_ROLL[0],
        target_pitch=TARGET_PITCH[0],
        target_yaw=TARGET_YAW[0],
        goal_offset=PUSH_GOAL_OFFSET,
        home_joint_names=list(ARM_JOINT_NAMES[:5]),
        home_success_distance=HOME_SUCCESS_DISTANCE,
        home_gate_sharpness=HOME_GATE_SHARPNESS,
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
class ActionsCfg:
    arm_action = mdp.ArmJointPositionWithOpenGripperActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=0.5,
        use_default_offset=True,
        preserve_order=False,
        gripper_joint_names=list(GRIPPER_JOINT_NAMES),
        gripper_open_position=0.0,
    )


@configclass
class ObservationsCfg:
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
            params={"robot_cfg": ROBOT_CFG, "object_cfg": TARGET_OBJECT_CFG},
        )
        cube_width = ObsTerm(func=mdp.cube_width, params={"width": CUBE_WIDTH})
        current_ee_pose = ObsTerm(
            func=mdp.current_ee_pose_b,
            params={"robot_cfg": ROBOT_CFG, "frame_cfg": EE_FRAME_CFG},
        )
        sweep_goal_position = ObsTerm(
            func=mdp.sweep_goal_position_b,
            params={"command_name": "ee_pose", "robot_cfg": ROBOT_CFG},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": ARM_RESET_CFG,
        },
    )


@configclass
class RewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.tcp_position_command_reward_exp,
        weight=REACH_REWARD_WEIGHT,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.align_ee_to_shelf,
        weight=ORIENTATION_REWARD_WEIGHT,
        params={
            "shelf_cfg": SHELF_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": ARM_CFG},
    )
    shelf_collision = RewTerm(
        func=mdp.shelf_collision,
        weight=SHELF_COLLISION_WEIGHT,
        params={
            "sensor_name": "shelf_floor_contact",
            "force_threshold": SHELF_CONTACT_FORCE_THRESHOLD,
            "surface_heights": SHELF_FLOOR_SURFACE_HEIGHTS,
            "surface_tolerance": SHELF_FLOOR_SURFACE_TOLERANCE,
            "x_bounds": SHELF_FLOOR_X_BOUNDS,
            "y_bounds": SHELF_FLOOR_Y_BOUNDS,
            "shelf_cfg": SHELF_CFG,
        },
    )
    pushing_target = RewTerm(
        func=mdp.PushingTargetDeltaReward,
        weight=PUSH_REWARD_WEIGHT,
        params={
            "command_name": "ee_pose",
            "gate_distance": SWEEP_GATE_DISTANCE,
            "goal_distance": PUSH_GOAL_OFFSET[1],
            "velocity_reward_min": PUSH_VELOCITY_REWARD_MIN,
            "velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN,
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": EE_FRAME_CFG,
            "wrist_frame_cfg": WRIST_FRAME_CFG,
        },
    )
    homing_after_sweep = RewTerm(
        func=mdp.homing_reward,
        weight=HOME_REWARD_WEIGHT,
        params={
            "command_name": "ee_pose",
            "success_distance": HOME_SUCCESS_DISTANCE,
            "gate_sharpness": HOME_GATE_SHARPNESS,
            "object_cfg": TARGET_OBJECT_CFG,
            "asset_cfg": HOME_ARM_CFG,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_drop = DoneTerm(
        func=mdp.cube_below_minimum_height,
        time_out=False,
        params={
            "minimum_center_height": CUBE_DROP_MIN_CENTER_HEIGHT,
            "object_cfg": TARGET_OBJECT_CFG,
        },
    )
    cube_tilt = DoneTerm(
        func=mdp.cube_excessive_tilt,
        time_out=False,
        params={
            "maximum_tilt": CUBE_MAX_TILT_RADIANS,
            "object_cfg": TARGET_OBJECT_CFG,
        },
    )
    cube_too_fast = DoneTerm(
        func=mdp.cube_too_fast,
        time_out=False,
        params={
            "maximum_linear_speed": CUBE_MAX_LINEAR_SPEED,
            "object_cfg": TARGET_OBJECT_CFG,
        },
    )
    arm_joint_too_fast = DoneTerm(
        func=mdp.arm_joint_too_fast,
        time_out=False,
        params={
            "maximum_joint_speed": ARM_MAX_JOINT_SPEED,
            "asset_cfg": ARM_CFG,
        },
    )


@configclass
class ShelfCubeSweepV5EnvCfg(ManagerBasedRLEnvCfg):
    """Complete v5 MDP with no dependency on another sweep task package."""

    scene: ShelfCubeSweepV5SceneCfg = ShelfCubeSweepV5SceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        episode_duration = MAX_EPISODE_STEPS * self.decimation * self.sim.dt
        self.episode_length_s = math.nextafter(episode_duration, 0.0)
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=False, sim_device=self.sim.device
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False, sim_device=self.sim.device
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False, sim_device=self.sim.device
                ),
            }
        )


@configclass
class ShelfCubeSweepV5PlayEnvCfg(ShelfCubeSweepV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
