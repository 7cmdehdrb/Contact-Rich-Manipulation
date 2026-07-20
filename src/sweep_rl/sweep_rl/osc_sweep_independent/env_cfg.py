"""Standalone UR5e OSC sweep, park, and Home-return environment."""

from __future__ import annotations

import math

import torch

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

ARM_CFG = SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)
EEF_CFG = SceneEntityCfg("robot", body_names=[EEF_CENTER_BODY_NAME])
FT_CFG = SceneEntityCfg("robot", body_names=[FT_SENSOR_BODY_NAME])
TARGET_CFG = SceneEntityCfg("target_object")
PAD_SENSORS = ("left_contact", "right_contact")


@configclass
class IndependentSweepSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenTable",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.90, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.003, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.48, 0.36, 0.24)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.75)),
    )
    robot = make_ur5e_robotiq_ft_cfg()
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
            pos=(0.50, 0.0, 0.805), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    left_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_CONTACT_BODY_NAME}",
        update_period=0.0,
        history_length=0,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetCube"],
        debug_vis=False,
    )
    right_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_CONTACT_BODY_NAME}",
        update_period=0.0,
        history_length=0,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=8,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetCube"],
        debug_vis=False,
    )
    target_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_contact_points=False,
        max_contact_data_count_per_prim=32,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        debug_vis=False,
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
        workspace_x_range=(0.18, 0.82),
        workspace_y_range=(-0.36, 0.36),
        workspace_boundary_margin=0.015,
        endpoint_threshold=0.025,
        speed_threshold=0.020,
        goal_dwell_time=0.30,
        debug_vis=False,
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
            noise=Unoise(
                n_min=torch.tensor([-0.5, -0.5, -0.5, -0.02, -0.02, -0.02]),
                n_max=torch.tensor([0.5, 0.5, 0.5, 0.02, 0.02, 0.02]),
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
            noise=Unoise(
                n_min=torch.tensor([-0.003, -0.003, -0.003, -0.02, -0.02, -0.02]),
                n_max=torch.tensor([0.003, 0.003, 0.003, 0.02, 0.02, 0.02]),
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
    reset_target = EventTerm(
        func=mdp.reset_variable_size_target,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.14, 0.14),
                "yaw": (-math.pi, math.pi),
            },
            "table_top_height": 0.775,
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
            "contact_sensor_name": "target_robot_contact",
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
            "contact_sensor_name": "target_robot_contact",
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
            "sensor_name": "target_robot_contact",
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
            "minimum_height": 0.76,
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
        self.viewer.eye = (2.2, 2.2, 1.8)
        self.viewer.lookat = (0.45, 0.0, 0.80)
