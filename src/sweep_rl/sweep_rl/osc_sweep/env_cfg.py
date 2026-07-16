"""Manager-based UR5e variable-stiffness OSC sweep environment."""

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

from . import mdp
from .assets import (
    ARM_JOINT_NAMES,
    EEF_CENTER_BODY_NAME,
    FT_SENSOR_BODY_NAME,
    LEFT_CONTACT_BODY_NAME,
    RIGHT_CONTACT_BODY_NAME,
    make_ur5e_robotiq_ft_cfg,
)

ARM_ENTITY_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(ARM_JOINT_NAMES),
    preserve_order=True,
)
EEF_ENTITY_CFG = SceneEntityCfg(
    "robot",
    body_names=[EEF_CENTER_BODY_NAME],
)
FT_ENTITY_CFG = SceneEntityCfg(
    "robot",
    body_names=[FT_SENSOR_BODY_NAME],
)


@configclass
class OscSweepSceneCfg(InteractiveSceneCfg):
    """Open table, assembled robot, target cube, and filtered contact sensors."""

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
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.805),
            rot=(1.0, 0.0, 0.0, 0.0),
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

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=2500.0
        ),
    )


@configclass
class CommandsCfg:
    desired_motion = mdp.SweepMotionCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.10, 0.22),
        force_range=(8.0, 25.0),
        force_tolerance_range=(3.0, 6.0),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """The policy controls only six stiffness and six temporal pose values."""

    arm_action = mdp.SweepOperationalSpaceActionCfg(
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
    )


@configclass
class ObservationsCfg:
    """62-D observation in the exact order defined by the task specification."""

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
        ft_sensor = ObsTerm(
            func=mdp.virtual_ft_wrench_b,
            params={"asset_cfg": FT_ENTITY_CFG},
        )
        contact_point = ObsTerm(
            func=mdp.target_contact_point_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "sensor_names": ("left_contact", "right_contact"),
                "force_threshold": 0.25,
            },
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
class EventCfg:
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
            "asset_cfg": ARM_ENTITY_CFG,
        },
    )
    reset_target = EventTerm(
        func=mdp.reset_target_object,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward direction, requested endpoint, and target contact force."""

    reaching = RewTerm(
        func=mdp.reaching_precontact_pose,
        weight=1.5,
        params={
            "command_name": "desired_motion",
            "std": 0.25,
            "stand_off": 0.065,
            "eef_cfg": EEF_ENTITY_CFG,
        },
    )
    target_contact = RewTerm(
        func=mdp.target_contact_bonus,
        weight=0.5,
        params={"sensor_names": ("left_contact", "right_contact")},
    )
    force_tracking = RewTerm(
        func=mdp.target_force_tracking,
        weight=4.0,
        params={
            "command_name": "desired_motion",
            "sensor_names": ("left_contact", "right_contact"),
        },
    )
    velocity_progress = RewTerm(
        func=mdp.object_velocity_along_direction,
        weight=4.0,
        params={
            "command_name": "desired_motion",
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    normalized_progress = RewTerm(
        func=mdp.normalized_progress,
        weight=1.0,
        params={"command_name": "desired_motion"},
    )
    direction_alignment = RewTerm(
        func=mdp.direction_alignment,
        weight=1.5,
        params={"command_name": "desired_motion"},
    )
    endpoint_tracking = RewTerm(
        func=mdp.endpoint_tracking,
        weight=6.0,
        params={
            "command_name": "desired_motion",
            "std": 0.035,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    success = RewTerm(
        func=mdp.sweep_success_bonus,
        weight=10.0,
        params={
            "command_name": "desired_motion",
            "endpoint_threshold": 0.025,
            "lateral_threshold": 0.12,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    lateral_error = RewTerm(
        func=mdp.lateral_displacement,
        weight=-2.0,
        params={"command_name": "desired_motion"},
    )
    overshoot = RewTerm(
        func=mdp.overshoot_penalty,
        weight=-4.0,
        params={"command_name": "desired_motion"},
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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.target_reached,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "endpoint_threshold": 0.025,
            "lateral_threshold": 0.12,
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
class CurriculumCfg:
    """No curriculum is required for the first clean baseline."""

    pass


@configclass
class UR5eOscSweepEnvCfg(ManagerBasedRLEnvCfg):
    """Vectorized variable-stiffness OSC sweep training environment."""

    scene: OscSweepSceneCfg = OscSweepSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
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


@configclass
class UR5eOscSweepEnvCfg_PLAY(UR5eOscSweepEnvCfg):
    """Small deterministic scene for inspection and policy playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.commands.desired_motion.direction_angle_range = (
            math.pi / 2,
            math.pi / 2,
        )
        self.commands.desired_motion.distance_range = (0.16, 0.16)
        self.commands.desired_motion.force_range = (15.0, 15.0)
        self.commands.desired_motion.force_tolerance_range = (4.0, 4.0)
