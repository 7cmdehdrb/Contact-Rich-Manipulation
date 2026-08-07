"""Fixed +Y Sweep-Policy environment with one procedural Cube."""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp

TASK_ID = "Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0"

TARGET_SIDE = 0.06
TARGET_HEIGHT = 0.12
TARGET_MASS = 0.50
SHELF_POSITION = (-0.65, 0.0, 0.0)
SHELF_SURFACE_HEIGHT = 1.05
TARGET_CENTER_HEIGHT = SHELF_SURFACE_HEIGHT + 0.5 * TARGET_HEIGHT
GOAL_OFFSET = (0.0, 0.18, 0.0)
POSITION_JITTER = 0.02

# The Fixed Sweep-Policy was trained with this already-assembled UR5e/Robotiq
# asset.  Reusing it is important: the standalone Isaac Sim Robotiq has a
# different link coordinate system, joint set, and collision representation.
DEFAULT_SWEEP_POLICY_ROBOT_USD_PATH = (
    "/home/min/Desktop/Shelf_USD/Robots/UR5e/" "Collected_UR5e_v4/UR5e_v4.usd"
)
ROBOT_BASE_BODY_PATH = "base_link"
GRIPPER_BASE_BODY_PATH = "robotiq_base_link"

EE_FORWARD_OFFSET = 0.130
FINGER_CONTACT_LATERAL_OFFSET = 0.070
WRIST_BACK_OFFSET = -0.140

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Original six YAML positions shifted +0.05 m toward the manipulator, with z
# converted from asset-base height to Cuboid center.
TARGET_SLOTS = (
    (-0.70, -0.20, TARGET_CENTER_HEIGHT),
    (-0.70, 0.00, TARGET_CENTER_HEIGHT),
    (-0.70, 0.20, TARGET_CENTER_HEIGHT),
    (-0.55, -0.20, TARGET_CENTER_HEIGHT),
    (-0.55, 0.00, TARGET_CENTER_HEIGHT),
    (-0.55, 0.20, TARGET_CENTER_HEIGHT),
)

TARGET_CFG = SceneEntityCfg("target_object")
ROBOT_CFG = SceneEntityCfg("robot")

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


def make_sweep_policy_robot_cfg() -> ArticulationCfg:
    """Return the original Fixed Sweep-Policy UR5e/Robotiq articulation."""

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEP_POLICY_ROBOT_USD_PATH",
                DEFAULT_SWEEP_POLICY_ROBOT_USD_PATH,
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
class SweepPolicyCubeSceneCfg(InteractiveSceneCfg):
    """Ground, shelf, UR5e/Robotiq, one Cube, and light."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEP_SHELF_USD_PATH",
                "omniverse://192.168.0.13/Library/Shelf/Arena/"
                "Collected_speedrack_shape/speedrack_shape.usd",
            ),
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
    robot = make_sweep_policy_robot_cfg()
    ee_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="end_effector",
                offset=OffsetCfg(pos=(EE_FORWARD_OFFSET, 0.0, 0.0)),
            )
        ],
    )
    finger_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="left_finger",
                offset=OffsetCfg(
                    pos=(
                        EE_FORWARD_OFFSET,
                        FINGER_CONTACT_LATERAL_OFFSET,
                        0.0,
                    )
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="right_finger",
                offset=OffsetCfg(
                    pos=(
                        EE_FORWARD_OFFSET,
                        -FINGER_CONTACT_LATERAL_OFFSET,
                        0.0,
                    )
                ),
            ),
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
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_SIDE, TARGET_SIDE, TARGET_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_SLOTS[4], rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    target_goal_pos = mdp.FixedObjectGoalCommandCfg(
        robot_name="robot",
        object_name="target_object",
        goal_offset=GOAL_OFFSET,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Six arm commands; the gripper is held open by its implicit PD actuator."""

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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_arm)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel_arm)
        actions = ObsTerm(func=base_mdp.last_action)
        target_obs_state = ObsTerm(func=mdp.target_position_b)
        target_obj_width = ObsTerm(
            func=mdp.target_width,
            params={"width": TARGET_SIDE},
        )
        ee_pose = ObsTerm(func=mdp.end_effector_pose_b)
        goal_pos = ObsTerm(
            func=mdp.goal_position_b,
            params={"command_name": "target_goal_pos"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    object_spawn = EventTerm(
        func=mdp.reset_target_from_slots,
        mode="reset",
        params={
            "asset_cfg": TARGET_CFG,
            "slots": TARGET_SLOTS,
            "position_jitter": POSITION_JITTER,
            "center_height": TARGET_CENTER_HEIGHT,
        },
    )


@configclass
class RewardsCfg:
    """Reaching-only test-bench rewards."""

    end_effector_position_tracking = RewTerm(
        func=mdp.reaching_position_command_error,
        weight=-0.2,
        params={
            "command_name": "target_goal_pos",
            "target_width": TARGET_SIDE,
        },
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.reaching_position_command_error_tanh,
        weight=0.1,
        params={
            "std": 0.1,
            "command_name": "target_goal_pos",
            "target_width": TARGET_SIDE,
        },
    )

    # Penalize only the task-relevant axis error: EE y-axis should match the
    # shelf z-axis. Rotation about that aligned axis remains unconstrained.
    orientation = RewTerm(
        func=mdp.ee_y_shelf_z_orientation_error,
        weight=-0.1,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_arm,
        weight=-0.0001,
    )

    # Disabled for the Reaching-only test bench. Kept here for restoration.
    # shelf_collision = RewTerm(func=mdp.ShelfCollisionPenalty, weight=-0.1)
    # reaching = RewTerm(
    #     func=mdp.reward_for_hand_reaching,
    #     weight=5.0,
    #     params={
    #         "command_name": "target_goal_pos",
    #         "target_width": TARGET_SIDE,
    #     },
    # )
    # sweeping_object = RewTerm(
    #     func=mdp.pushing_target,
    #     weight=6.0,
    #     params={
    #         "command_name": "target_goal_pos",
    #         "target_width": TARGET_SIDE,
    #     },
    # )
    # homing_after_sweep = RewTerm(
    #     func=mdp.homing_reward,
    #     weight=12.0,
    #     params={"command_name": "target_goal_pos"},
    # )


@configclass
class TerminationsCfg:
    """Reaching-only test-bench termination terms."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Disabled for the Reaching-only test bench. Kept here for restoration.
    # object_drop = DoneTerm(
    #     func=mdp.drop_object_termination,
    #     time_out=False,
    #     params={"height_condition": 1.04, "rotation_condition": 0.9},
    # )
    # push_fast = DoneTerm(
    #     func=mdp.push_fast_termination,
    #     time_out=False,
    #     params={"speed_condition": 0.3},
    # )
    # shelf_collision = DoneTerm(
    #     func=mdp.shelf_collision_termination,
    #     time_out=False,
    #     params={"threshold": 0.1},
    # )
    # hand_velocity = DoneTerm(
    #     func=mdp.hand_velocity_termination,
    #     time_out=False,
    #     params={"threshold": 2.0},
    # )


@configclass
class SweepPolicyCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: SweepPolicyCubeSceneCfg = SweepPolicyCubeSceneCfg(
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        debug_vis = os.environ.get("SWEEP_POLICY_CUBE_DEBUG_VIS", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.commands.target_goal_pos.debug_vis = debug_vis
        self.scene.ee_frame.debug_vis = debug_vis
        self.scene.finger_frame.debug_vis = debug_vis
        self.scene.wrist_frame.debug_vis = debug_vis
