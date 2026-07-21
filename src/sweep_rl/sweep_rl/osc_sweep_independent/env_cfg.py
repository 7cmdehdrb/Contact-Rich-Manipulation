"""Standalone UR5e OSC sweep, park, and Home-return environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..osc_sweep.assets import (
    ARM_JOINT_NAMES,
    EEF_CENTER_BODY_NAME,
    FT_SENSOR_BODY_NAME,
    LEFT_CONTACT_BODY_NAME,
    RIGHT_CONTACT_BODY_NAME,
    make_ur5e_robotiq_ft_cfg,
)
from . import mdp

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
EEF_CFG = SceneEntityCfg("robot", body_names=[EEF_CENTER_BODY_NAME])
FT_CFG = SceneEntityCfg("robot", body_names=[FT_SENSOR_BODY_NAME])
TARGET_CFG = SceneEntityCfg("target_object")
PAD_SENSORS = ("left_contact", "right_contact")
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
    LEFT_CONTACT_BODY_NAME,
    RIGHT_CONTACT_BODY_NAME,
)
ROBOT_CONTACT_SENSORS = (
    "base_contact",
    "shoulder_contact",
    "upper_arm_contact",
    "forearm_contact",
    "wrist_1_contact",
    "wrist_2_contact",
    "wrist_3_contact",
    "gripper_base_contact",
    "left_outer_knuckle_contact",
    "left_outer_finger_contact",
    "left_inner_finger_contact",
    "left_inner_knuckle_contact",
    "right_outer_knuckle_contact",
    "right_outer_finger_contact",
    "right_inner_finger_contact",
    "right_inner_knuckle_contact",
    *PAD_SENSORS,
)
ROBOT_CONTACT_FILTERS = (
    "{ENV_REGEX_NS}/TargetCube",
    "{ENV_REGEX_NS}/Shelf/rack",
    *(f"{{ENV_REGEX_NS}}/Robot/{body_path}" for body_path in ROBOT_CONTACT_BODY_PATHS),
)
TARGET_CONTACT_FILTER_INDEX = 0
SHELF_CONTACT_FILTER_INDEX = 1
SELF_CONTACT_FILTER_START_INDEX = 2
# These pairs overlap by construction at the shelf-policy neutral gripper pose or
# are directly connected by a fixed/revolute joint.  All other UR/gripper pairs,
# including non-adjacent UR-to-gripper contacts, remain termination candidates.
SELF_COLLISION_EXCLUDED_PAIRS = (
    ("wrist_3_contact", "gripper_base_contact"),
    ("gripper_base_contact", "left_inner_knuckle_contact"),
    ("gripper_base_contact", "right_inner_knuckle_contact"),
    ("left_inner_finger_contact", "right_contact"),
    ("right_inner_finger_contact", "left_contact"),
)

# Bounds measured from the middle shelf board in the composed USD.  The command
# operates in the 180-degree-rotated robot-root frame, so the board's world X
# interval [-0.90, -0.50] becomes root-frame X [0.50, 0.90].
SHELF_USD_PATH = "omniverse://192.168.0.13/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd"
SHELF_POSITION = (-0.7, 0.0, 0.0)
SHELF_SURFACE_HEIGHT = 1.05
SHELF_WORKSPACE_X_RANGE = (0.50, 0.90)
SHELF_WORKSPACE_Y_RANGE = (-0.50, 0.50)


def make_shelf_ur5e_robotiq_ft_cfg():
    """Place the sensor-equipped sweep robot as in the shelf environment."""
    robot_cfg = make_ur5e_robotiq_ft_cfg()
    robot_cfg.init_state.pos = (0.0, 0.0, 0.79505)
    robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot_cfg.init_state.joint_pos = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -2.2,
        "elbow_joint": 2.2,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 1.57,
        "wrist_3_joint": 0.785,
        ".*(finger|knuckle).*": 0.0,
    }
    robot_cfg.init_state.joint_vel = {".*": 0.0}
    robot_cfg.spawn.articulation_props.enabled_self_collisions = True
    return robot_cfg


def make_robot_body_contact_sensor(
    body_path: str,
    *,
    track_pose: bool = False,
    track_contact_points: bool = False,
) -> ContactSensorCfg:
    """Create a one-body sensor with TargetCube and Shelf contact filters."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_path}",
        update_period=0.0,
        history_length=0,
        track_pose=track_pose,
        track_contact_points=track_contact_points,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=list(ROBOT_CONTACT_FILTERS),
        debug_vis=False,
    )


@configclass
class IndependentSweepSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=SHELF_USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SHELF_POSITION, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )
    robot = make_shelf_ur5e_robotiq_ft_cfg()
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.003, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65, dynamic_friction=0.45, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.60, 0.0, 1.05), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    base_contact = make_robot_body_contact_sensor("base_link")
    shoulder_contact = make_robot_body_contact_sensor("shoulder_link")
    upper_arm_contact = make_robot_body_contact_sensor("upper_arm_link")
    forearm_contact = make_robot_body_contact_sensor("forearm_link")
    wrist_1_contact = make_robot_body_contact_sensor("wrist_1_link")
    wrist_2_contact = make_robot_body_contact_sensor("wrist_2_link")
    wrist_3_contact = make_robot_body_contact_sensor("wrist_3_link")
    gripper_base_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/base_link"
    )
    left_outer_knuckle_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/left_outer_knuckle"
    )
    left_outer_finger_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/left_outer_finger"
    )
    left_inner_finger_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/left_inner_finger"
    )
    left_inner_knuckle_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/left_inner_knuckle"
    )
    right_outer_knuckle_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/right_outer_knuckle"
    )
    right_outer_finger_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/right_outer_finger"
    )
    right_inner_finger_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/right_inner_finger"
    )
    right_inner_knuckle_contact = make_robot_body_contact_sensor(
        "Robotiq2F85/Robotiq_2F_85/right_inner_knuckle"
    )
    left_contact = make_robot_body_contact_sensor(
        LEFT_CONTACT_BODY_NAME,
        track_pose=True,
        track_contact_points=True,
    )
    right_contact = make_robot_body_contact_sensor(
        RIGHT_CONTACT_BODY_NAME,
        track_pose=True,
        track_contact_points=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    desired_motion = mdp.FeasibleSweepHomeCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.12, 0.35),
        target_speed_range=(0.04, 0.12),
        workspace_x_range=SHELF_WORKSPACE_X_RANGE,
        workspace_y_range=SHELF_WORKSPACE_Y_RANGE,
        workspace_boundary_margin=0.015,
        endpoint_threshold=0.025,
        speed_threshold=0.020,
        goal_dwell_time=0.30,
        contact_sensor_names=PAD_SENSORS,
        contact_force_threshold=0.25,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    arm_action = mdp.IndependentSweepOscActionCfg(
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
            motion_stiffness_task=(120.0, 120.0, 120.0, 35.0, 35.0, 35.0),
            motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            motion_stiffness_limits_task=(20.0, 300.0),
            nullspace_control="none",
        ),
        position_scale=0.025,
        orientation_scale=0.12,
        stiffness_scale=1.0,
        nullspace_joint_pos_target="none",
        effort_limit_scale=0.9,
        stiffness_calibration_range=(0.95, 1.05),
        damping_calibration_range=(0.95, 1.05),
        effort_calibration_range=(0.97, 1.03),
        gripper_joint_names=[".*(finger|knuckle).*"],
        gripper_open_position=0.0,
    )


@configclass
class ObservationsCfg:
    """56-D policy observation; live object pose/velocity remain privileged."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": ARM_CFG},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": ARM_CFG},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        eef_pose = ObsTerm(func=mdp.end_effector_pose_b, params={"asset_cfg": EEF_CFG})
        ft_sensor = ObsTerm(
            func=mdp.virtual_ft_wrench_b,
            params={"asset_cfg": FT_CFG},
            noise=mdp.VectorUniformNoiseCfg(
                n_min=[-0.5, -0.5, -0.5, -0.02, -0.02, -0.02],
                n_max=[0.5, 0.5, 0.5, 0.02, 0.02, 0.02],
            ),
        )
        contact_point = ObsTerm(
            func=mdp.target_contact_point_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "sensor_names": PAD_SENSORS,
                "force_threshold": 0.25,
            },
            noise=mdp.MaskedUniformNoiseCfg(n_min=-0.002, n_max=0.002),
        )
        initial_target_pose = ObsTerm(
            func=mdp.initial_target_pose_b,
            params={"command_name": "desired_motion"},
            noise=mdp.VectorUniformNoiseCfg(
                n_min=[-0.003, -0.003, -0.003, -0.02, -0.02, -0.02],
                n_max=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02],
            ),
        )
        desired_motion = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "desired_motion"}
        )
        task_phase = ObsTerm(
            func=mdp.task_phase, params={"command_name": "desired_motion"}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    create_position_visualizers = EventTerm(
        func=mdp.create_sweep_position_visualizers,
        mode="prestartup",
    )
    randomize_target_size = EventTerm(
        func=mdp.randomize_target_cube_size,
        mode="prestartup",
        params={
            "size_range": (0.04, 0.08),
            "base_size": 0.06,
            "asset_cfg": TARGET_CFG,
        },
    )
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.04, 0.04),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": ARM_CFG,
        },
    )
    randomize_target_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": TARGET_CFG,
            "mass_distribution_params": (0.25, 2.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_target_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": TARGET_CFG,
            "static_friction_range": (0.40, 1.10),
            "dynamic_friction_range": (0.25, 0.90),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    randomize_shelf_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("shelf"),
            "static_friction_range": (0.40, 1.10),
            "dynamic_friction_range": (0.25, 0.90),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_target = EventTerm(
        func=mdp.reset_variable_size_target,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.14, 0.14),
                "yaw": (-math.pi, math.pi),
            },
            "table_top_height": SHELF_SURFACE_HEIGHT,
            "asset_cfg": TARGET_CFG,
        },
    )


@configclass
class RewardsCfg:
    """Only the four user-approved reward terms."""

    reaching = RewTerm(
        func=mdp.reaching,
        weight=1.0,
        params={
            "command_name": "desired_motion",
            "distance_std": 0.12,
            "surface_clearance": 0.008,
            "table_side_pad_offset": 0.055,
            "sensor_names": PAD_SENSORS,
            "eef_cfg": EEF_CFG,
            "object_cfg": TARGET_CFG,
        },
    )
    contact = RewTerm(
        func=mdp.contact,
        weight=1.5,
        params={
            "command_name": "desired_motion",
            "sensor_names": PAD_SENSORS,
            "force_threshold": 0.25,
        },
    )
    push = RewTerm(
        func=mdp.push,
        weight=2.0,
        params={
            "command_name": "desired_motion",
            "sensor_names": PAD_SENSORS,
            "velocity_std": 0.035,
            "acceleration_distance": 0.030,
            "stopping_distance": 0.050,
            "initial_speed_fraction": 0.25,
            "endpoint_std": 0.035,
            "stopped_speed_std": 0.025,
            "object_cfg": TARGET_CFG,
        },
    )
    home_return = RewTerm(
        func=mdp.home_return,
        weight=2.0,
        params={
            "command_name": "desired_motion",
            "joint_std": 0.35,
            "joint_error_scale": 0.75,
            "contact_sensor_name": ROBOT_CONTACT_SENSORS,
            "contact_force_threshold": 0.25,
            "displacement_scale": 0.010,
            "asset_cfg": ARM_CFG,
            "object_cfg": TARGET_CFG,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.HomeAfterSweepSuccess,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "joint_position_threshold": 0.12,
            "joint_speed_threshold": 0.15,
            "endpoint_threshold": 0.030,
            "object_speed_threshold": 0.025,
            "object_displacement_threshold": 0.010,
            "dwell_time": 0.25,
            "contact_sensor_name": ROBOT_CONTACT_SENSORS,
            "contact_force_threshold": 0.25,
            "asset_cfg": ARM_CFG,
            "object_cfg": TARGET_CFG,
        },
    )
    contact_lost = DoneTerm(
        func=mdp.ContactLostTooLong,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "sensor_names": PAD_SENSORS,
            "force_threshold": 0.25,
            "maximum_loss_time": 0.75,
        },
    )
    object_inside_gripper = DoneTerm(
        func=mdp.object_inside_gripper,
        time_out=False,
        params={
            "center_half_extents": (0.040, 0.040, 0.058),
            "eef_cfg": EEF_CFG,
            "object_cfg": TARGET_CFG,
        },
    )
    home_contact = DoneTerm(
        func=mdp.HomeContactAfterRelease,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "sensor_name": ROBOT_CONTACT_SENSORS,
            "force_threshold": 0.25,
            "release_grace_time": 0.30,
        },
    )
    object_disturbed_home = DoneTerm(
        func=mdp.object_disturbed_during_home,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "displacement_threshold": 0.015,
            "speed_threshold": 0.10,
            "object_cfg": TARGET_CFG,
        },
    )
    target_invalid_pose = DoneTerm(
        func=mdp.target_invalid_pose,
        time_out=False,
        params={
            "minimum_height": 1.04,
            "maximum_tilt": 0.80,
            "object_cfg": TARGET_CFG,
        },
    )
    excessive_wrench = DoneTerm(
        func=mdp.excessive_ft_wrench,
        time_out=False,
        params={"force_limit": 100.0, "torque_limit": 15.0, "asset_cfg": FT_CFG},
    )
    arm_speed = DoneTerm(
        func=mdp.arm_joint_speed_limit,
        time_out=False,
        params={"maximum_speed": 6.5, "asset_cfg": ARM_CFG},
    )
    shelf_collision = DoneTerm(
        func=mdp.robot_shelf_collision,
        time_out=False,
        params={
            "sensor_names": ROBOT_CONTACT_SENSORS,
            "shelf_filter_index": SHELF_CONTACT_FILTER_INDEX,
            "force_threshold": 0.1,
        },
    )
    self_collision = DoneTerm(
        func=mdp.robot_self_collision,
        time_out=False,
        params={
            "sensor_names": ROBOT_CONTACT_SENSORS,
            "self_filter_start_index": SELF_CONTACT_FILTER_START_INDEX,
            "excluded_pairs": SELF_COLLISION_EXCLUDED_PAIRS,
            "force_threshold": 0.1,
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class UR5eOscSweepIndependentEnvCfg(ManagerBasedRLEnvCfg):
    """Direct, non-inheriting environment configuration."""

    scene: IndependentSweepSceneCfg = IndependentSweepSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.viewer.eye = (1.6, 2.2, 1.8)
        self.viewer.lookat = (-0.65, 0.0, 1.02)
