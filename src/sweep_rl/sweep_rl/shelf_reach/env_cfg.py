"""UR5e + Robotiq pose-reaching task inside the Sweep-Policy shelf."""

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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-v0"
PLAY_TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-Play-v0"

DEFAULT_ROBOT_USD_PATH = (
    "./asset/Shelf_USD/Robots/UR5e/Collected_UR5e_v4/.collect.mapping.json"
)
DEFAULT_SHELF_USD_PATH = (
    "omniverse://192.168.0.13/Library/Shelf/Arena/"
    "Collected_speedrack_shape/speedrack_shape.usd"
)

SHELF_POSITION = (-0.7, 0.0, 0.0)
ROBOT_BASE_BODY_PATH = "base_link"
GRIPPER_BASE_BODY_PATH = "robotiq_base_link"
TCP_FORWARD_OFFSET = 0.13

TARGET_POS_X = (0.55, 0.75)
TARGET_POS_Y = (-0.20, 0.20)
TARGET_POS_Z = (0.30, 0.50)
# The default Sweep-Policy posture puts the TCP x-axis along robot-base +X
# with an approximately +45-degree roll.  Add another 45 degrees so that the
# target TCP y-axis points exactly along robot-base +Z (toward the ceiling).
TARGET_ROLL = (math.pi / 2.0, math.pi / 2.0)
TARGET_PITCH = (0.0, 0.0)
TARGET_YAW = (0.0, 0.0)

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

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
ROBOT_CFG = SceneEntityCfg("robot")
EE_FRAME_CFG = SceneEntityCfg("ee_frame")


def make_sweep_policy_robot_cfg() -> ArticulationCfg:
    """Build the combined UR5e/Robotiq asset used by Sweep-Policy."""

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEP_POLICY_ROBOT_USD_PATH", DEFAULT_ROBOT_USD_PATH
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
                "wrist_3_joint": 0.785,
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
                    "shoulder_pan_joint": 3.14,
                    "shoulder_lift_joint": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_1_joint": 6.28,
                    "wrist_2_joint": 6.28,
                    "wrist_3_joint": 6.28,
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
class ShelfReachSceneCfg(InteractiveSceneCfg):
    """Sweep-Policy ground, shelf, combined robot, TCP frame, and light."""

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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SHELF_POSITION,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )
    robot = make_sweep_policy_robot_cfg()
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
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    ee_pose = mdp.TcpUniformPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.TcpUniformPoseCommandCfg.Ranges(
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
    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
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
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
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
            "asset_cfg": ARM_CFG,
        },
    )


@configclass
class RewardsCfg:
    end_effector_position_tracking = RewTerm(
        func=mdp.tcp_position_command_error,
        weight=-0.2,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.tcp_position_command_error_tanh,
        weight=0.1,
        params={
            "std": 0.1,
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.tcp_orientation_command_error,
        weight=-0.1,
        params={
            "command_name": "ee_pose",
            "robot_cfg": ROBOT_CFG,
            "frame_cfg": EE_FRAME_CFG,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": ARM_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={
            "term_name": "action_rate",
            "weight": -0.005,
            "num_steps": 4500,
        },
    )
    joint_vel = CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel",
            "weight": -0.001,
            "num_steps": 4500,
        },
    )


@configclass
class UR5eGripperShelfReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ShelfReachSceneCfg = ShelfReachSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
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
class UR5eGripperShelfReachEnvCfg_PLAY(UR5eGripperShelfReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
