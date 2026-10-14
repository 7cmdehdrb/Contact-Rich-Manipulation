"""Manager-based, single-stage UR5e shelf sweep environment."""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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

from . import mdp
from .assets import (
    ARM_JOINT_NAMES,
    GRIPPER_BASE_BODY_PATH,
    GRIPPER_JOINT_PATTERN,
    LEFT_FINGER_BODY_PATH,
    RIGHT_FINGER_BODY_PATH,
    ROBOT_BASE_BODY_PATH,
    WRIST_BODY_PATH,
    make_sensorless_robot_cfg,
)

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
TARGET_CFG = SceneEntityCfg("target_object")
SHELF_CFG = SceneEntityCfg("shelf")
FRAME_CFG = SceneEntityCfg("ee_frame")

ROBOT_CONTACT_BODY_PATHS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "Robotiq2F85/Robotiq_2F_85/base_link",
    "Robotiq2F85/Robotiq_2F_85/left_outer_knuckle",
    "Robotiq2F85/Robotiq_2F_85/left_outer_finger",
    "Robotiq2F85/Robotiq_2F_85/left_inner_finger",
    "Robotiq2F85/Robotiq_2F_85/left_inner_knuckle",
    "Robotiq2F85/Robotiq_2F_85/right_outer_knuckle",
    "Robotiq2F85/Robotiq_2F_85/right_outer_finger",
    "Robotiq2F85/Robotiq_2F_85/right_inner_finger",
    "Robotiq2F85/Robotiq_2F_85/right_inner_knuckle",
)
GRIPPER_CONTACT_BODY_PATHS = ROBOT_CONTACT_BODY_PATHS[7:]

EEF_FRAME_INDEX = 0
LEFT_FINGER_FRAME_INDEX = 1
RIGHT_FINGER_FRAME_INDEX = 2
WRIST_FRAME_INDEX = 3

DEFAULT_SHELF_USD_PATH = (
    "omniverse://192.168.0.13/Library/Shelf/Arena/"
    "Collected_speedrack_shape/speedrack_shape.usd"
)
SHELF_POSITION = (-0.7, 0.0, 0.0)
SHELF_SURFACE_HEIGHT_OFFSET = 1.05
SHELF_TOP_X_BOUNDS = (-0.20, 0.20)
SHELF_TOP_Y_BOUNDS = (-0.50, 0.50)

TARGET_WIDTH = 0.06
TARGET_HEIGHT = 0.07
TARGET_MASS = 0.50
TARGET_MAXIMUM_HEIGHT_CHANGE = 0.5 * TARGET_HEIGHT
TARGET_INITIAL_POSITION = (
    -0.70,
    -0.10,
    SHELF_SURFACE_HEIGHT_OFFSET + 0.5 * TARGET_HEIGHT,
)
TARGET_X_RANGE = (-0.72, -0.68)
TARGET_Y_RANGE = (-0.12, -0.08)

SWEEP_DISTANCE = 0.40
TARGET_SPEED = 0.10
PRECONTACT_STANDOFF = 0.060
PUSH_STANDOFF = 0.040
EXTERNAL_FINGER_OFFSET = 0.055


@configclass
class BasicSweepSceneCfg(InteractiveSceneCfg):
    """The existing shelf workspace with one robot and one near-cubic target."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("SWEEP_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SHELF_POSITION, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )
    shelf_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/rack",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_contact_points=False,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{body_path}"
            for body_path in ROBOT_CONTACT_BODY_PATHS
        ],
        debug_vis=False,
    )
    robot = make_sensorless_robot_cfg()
    ee_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.16)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_FINGER_BODY_PATH}",
                name="left_finger",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_FINGER_BODY_PATH}",
                name="right_finger",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{WRIST_BODY_PATH}",
                name="wrist",
            ),
        ],
    )
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_WIDTH, TARGET_WIDTH, TARGET_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.45,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_INITIAL_POSITION,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    target_gripper_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_contact_points=False,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{body_path}"
            for body_path in GRIPPER_CONTACT_BODY_PATHS
        ],
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    target_goal = mdp.FixedSweepCommandCfg(
        robot_name="robot",
        object_name="target_object",
        shelf_name="shelf",
        frame_name="ee_frame",
        eef_frame_index=EEF_FRAME_INDEX,
        contact_frame_index=EEF_FRAME_INDEX,
        wrist_frame_index=WRIST_FRAME_INDEX,
        contact_sensor_name="target_gripper_contact",
        contact_force_threshold=0.25,
        precontact_standoff=PRECONTACT_STANDOFF,
        push_standoff=PUSH_STANDOFF,
        external_finger_offset=EXTERNAL_FINGER_OFFSET,
        sweep_distance=SWEEP_DISTANCE,
        target_speed=TARGET_SPEED,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    arm_action = mdp.OpenGripperRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=0.05,
        use_default_offset=False,
        preserve_order=True,
        gripper_joint_names=[GRIPPER_JOINT_PATTERN],
        gripper_open_position=0.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": ARM_CFG})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": ARM_CFG})
        last_action = ObsTerm(
            func=base_mdp.last_action, params={"action_name": "arm_action"}
        )
        target_object_state = ObsTerm(
            func=mdp.target_position_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": TARGET_CFG,
            },
        )
        goal_position = ObsTerm(
            func=mdp.goal_position_b,
            params={"command_name": "target_goal"},
        )
        eef_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "frame_cfg": FRAME_CFG,
                "frame_index": EEF_FRAME_INDEX,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": ARM_CFG})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": ARM_CFG})
        last_action = ObsTerm(
            func=base_mdp.last_action, params={"action_name": "arm_action"}
        )
        target_object_state = ObsTerm(
            func=mdp.target_position_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": TARGET_CFG,
            },
        )
        goal_position = ObsTerm(
            func=mdp.goal_position_b,
            params={"command_name": "target_goal"},
        )
        eef_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "frame_cfg": FRAME_CFG,
                "frame_index": EEF_FRAME_INDEX,
            },
        )
        target_linear_velocity = ObsTerm(
            func=mdp.target_linear_velocity_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": TARGET_CFG,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_target = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
            },
            "velocity_range": {},
            "asset_cfg": TARGET_CFG,
        },
    )


@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_vel_l2 = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": ARM_CFG},
    )
    shelf_collision = RewTerm(
        func=mdp.shelf_collision,
        weight=-2.0,
        params={
            "surface_height": SHELF_SURFACE_HEIGHT_OFFSET,
            "x_bounds": SHELF_TOP_X_BOUNDS,
            "y_bounds": SHELF_TOP_Y_BOUNDS,
            "finger_clearance": 0.02,
            "wrist_clearance": 0.08,
            "left_finger_frame_index": LEFT_FINGER_FRAME_INDEX,
            "right_finger_frame_index": RIGHT_FINGER_FRAME_INDEX,
            "wrist_frame_index": WRIST_FRAME_INDEX,
            "frame_cfg": FRAME_CFG,
            "shelf_cfg": SHELF_CFG,
        },
    )
    reward_for_hand_reaching = RewTerm(
        func=mdp.reward_for_hand_reaching,
        weight=2.0,
        params={
            "command_name": "target_goal",
            "distance_std": 0.25,
        },
    )
    align_ee_target = RewTerm(
        func=mdp.align_ee_target,
        weight=2.0,
        params={"command_name": "target_goal"},
    )
    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=6.0,
        params={
            "command_name": "target_goal",
            "contact_position_threshold": 0.04,
            "wrist_y_threshold": 0.04,
            "goal_threshold": 0.03,
            "distance_scale": 0.18,
            "velocity_reward_min": 0.05,
            "velocity_penalty_min": 0.10,
            "object_cfg": TARGET_CFG,
        },
    )
    target_contact = RewTerm(
        func=mdp.target_contact_bonus,
        weight=1.0,
        params={"command_name": "target_goal"},
    )
    contact_forward_progress = RewTerm(
        func=mdp.contact_forward_progress,
        weight=4.0,
        params={
            "command_name": "target_goal",
            "maximum_normalized_speed": 1.25,
            "object_cfg": TARGET_CFG,
        },
    )
    velocity_tracking = RewTerm(
        func=mdp.object_velocity_tracking,
        weight=8.0,
        params={
            "command_name": "target_goal",
            "std": 0.05,
            "object_cfg": TARGET_CFG,
        },
    )
    endpoint_error = RewTerm(
        func=mdp.normalized_endpoint_error,
        weight=-2.0,
        params={
            "command_name": "target_goal",
            "maximum_error": 2.0,
            "object_cfg": TARGET_CFG,
        },
    )
    stopped_at_goal = RewTerm(
        func=mdp.stopped_at_goal,
        weight=15.0,
        params={
            "command_name": "target_goal",
            "position_std": 0.03,
            "speed_std": 0.02,
            "object_cfg": TARGET_CFG,
        },
    )
    sweep_success = RewTerm(
        func=mdp.sweep_success_bonus,
        weight=30.0,
        params={
            "command_name": "target_goal",
            "endpoint_threshold": 0.03,
            "speed_threshold": 0.02,
            "object_cfg": TARGET_CFG,
        },
    )
    lateral_error = RewTerm(
        func=mdp.lateral_displacement,
        weight=-2.0,
        params={"command_name": "target_goal"},
    )
    overshoot = RewTerm(
        func=mdp.overshoot_penalty,
        weight=-4.0,
        params={"command_name": "target_goal"},
    )
    stall = RewTerm(
        func=mdp.object_stall_penalty,
        weight=-2.0,
        params={
            "command_name": "target_goal",
            "startup_grace_time": 2.0,
            "endpoint_threshold": 0.03,
            "minimum_speed_fraction": 0.50,
            "object_cfg": TARGET_CFG,
        },
    )
    midpoint_no_entry = RewTerm(
        func=mdp.MidpointNoEntryPenalty,
        weight=-10.0,
        params={
            "command_name": "target_goal",
            "contact_dwell_time": 0.10,
            "episode_fraction": 0.50,
        },
    )
    shelf_collision_failure = RewTerm(
        func=mdp.remaining_horizon_failure_penalty,
        weight=-8.0,
        params={
            "term_names": ("shelf_collision",),
            "minimum_penalty_time": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.target_reached_and_stopped,
        time_out=False,
        params={
            "command_name": "target_goal",
            "endpoint_threshold": 0.03,
            "speed_threshold": 0.02,
            "object_cfg": TARGET_CFG,
        },
    )
    object_drop = DoneTerm(
        func=mdp.object_drop,
        params={
            "command_name": "target_goal",
            "maximum_height_change": TARGET_MAXIMUM_HEIGHT_CHANGE,
            "object_cfg": TARGET_CFG,
        },
    )
    push_fast = DoneTerm(
        func=mdp.PushFastTermination,
        params={
            "initial_speed": 0.30,
            "intermediate_speed": 0.20,
            "final_speed": 0.15,
            "first_transition_step": 100_000,
            "second_transition_step": 250_000,
            "sustained_steps": 3,
            "object_cfg": TARGET_CFG,
        },
    )
    shelf_collision = DoneTerm(
        func=mdp.robot_shelf_collision,
        params={
            "sensor_name": "shelf_contact",
            "force_threshold": 0.1,
        },
    )


@configclass
class SweepBasicEnvCfg(ManagerBasedRLEnvCfg):
    scene: BasicSweepSceneCfg = BasicSweepSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=False,
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
        self.episode_length_s = 15.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.viewer.eye = (1.6, 2.2, 1.8)
        self.viewer.lookat = (-0.65, 0.0, 1.02)
        debug_vis = os.environ.get("SWEEP_BASIC_DEBUG_VIS", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.commands.target_goal.debug_vis = debug_vis
        self.scene.ee_frame.debug_vis = debug_vis
