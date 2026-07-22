"""Standalone shelf-sweep environment with an inline articulation F/T joint."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
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
from .asset_manifest import (
    OBJECT_NAMES,
    OBJECT_POSES,
    OBJECT_USD_PATHS,
    OBJECT_WIDTH_TUPLE,
    SHELF_USD_PATH,
)
from .assets import (
    ARM_JOINT_NAMES,
    EEF_CENTER_BODY_NAME,
    FT_SENSOR_BODY_NAME,
    make_shelf_force_robot_cfg,
)


COMMAND_NAME = "desired_motion"

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
FT_CFG = SceneEntityCfg("robot", body_names=[FT_SENSOR_BODY_NAME])
EEF_CFG = SceneEntityCfg("robot", body_names=[EEF_CENTER_BODY_NAME])
EE_FRAME_CFG = SceneEntityCfg("ee_frame")
FINGER_FRAME_CFG = SceneEntityCfg("finger_frame")
WRIST_FRAME_CFG = SceneEntityCfg("wrist_frame")
SHELF_CFG = SceneEntityCfg("shelf")


def _make_object_collection_cfg() -> RigidObjectCollectionCfg:
    rigid_objects = {}
    for name in OBJECT_NAMES:
        pose = OBJECT_POSES[name]
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pose[:3],
                rot=pose[3:7],
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=OBJECT_USD_PATHS[name],
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class ShelfForceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0,
        ),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHELF_USD_PATH,
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    robot = make_shelf_force_robot_cfg()
    object_collection = _make_object_collection_cfg()

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{EEF_CENTER_BODY_NAME}",
                name="end_effector",
            )
        ],
    )
    finger_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{EEF_CENTER_BODY_NAME}",
                name="left_finger_proxy",
                offset=OffsetCfg(pos=(0.0, 0.07, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{EEF_CENTER_BODY_NAME}",
                name="right_finger_proxy",
                offset=OffsetCfg(pos=(0.0, -0.07, 0.0)),
            ),
        ],
    )
    wrist_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{FT_SENSOR_BODY_NAME}",
                name="wrist",
            )
        ],
    )


@configclass
class CommandsCfg:
    desired_motion = mdp.ShelfSweepGoalCommandCfg(
        asset_name="object_collection",
        robot_name="robot",
        object_widths=OBJECT_WIDTH_TUPLE,
        sweep_distance=0.18,
        force_range=(8.0, 25.0),
        force_tolerance_range=(3.0, 6.0),
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    arm_action = mdp.ShelfOperationalSpaceActionCfg(
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
    )


@configclass
class ObservationsCfg:
    """Same 41-D task-space policy observation used by sweep_jh."""

    @configclass
    class PolicyCfg(ObsGroup):
        eef_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={"asset_cfg": EEF_CFG},
        )
        eef_twist = ObsTerm(
            func=mdp.end_effector_twist_b,
            params={"asset_cfg": EEF_CFG},
        )
        ft_sensor = ObsTerm(
            func=mdp.virtual_ft_wrench_b,
            params={"asset_cfg": FT_CFG},
        )
        initial_target_pose = ObsTerm(
            func=mdp.initial_target_pose_b,
            params={"command_name": COMMAND_NAME},
        )
        desired_motion = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    object_spawn = EventTerm(
        func=mdp.randomize_shelf_scene,
        mode="reset",
        params={
            "command_name": COMMAND_NAME,
            "ceiling_height": 1.8,
        },
    )


@configclass
class RewardsCfg:
    """Six compact task-level objectives for Cartesian force sweeping."""

    approach_error = RewTerm(
        func=mdp.approach_error,
        weight=-1.0,
        params={
            "command_name": COMMAND_NAME,
            "shelf_cfg": SHELF_CFG,
            "ee_frame_cfg": EE_FRAME_CFG,
            "ft_cfg": FT_CFG,
            "position_scale": 0.10,
            "near_distance": 0.02,
            "far_distance": 0.10,
            "force_low": 1.0,
            "force_high": 4.0,
            "position_fraction": 0.7,
            "x_offset": -0.02,
            "z_offset": 0.09,
        },
    )
    force_control_error = RewTerm(
        func=mdp.ForceControlError,
        weight=-2.0,
        params={
            "command_name": COMMAND_NAME,
            "ft_cfg": FT_CFG,
            "ee_frame_cfg": EE_FRAME_CFG,
            "near_distance": 0.02,
            "far_distance": 0.10,
            "tangential_scale": 25.0,
            "tangential_fraction": 0.3,
            "x_offset": -0.02,
            "z_offset": 0.09,
        },
    )
    sweep_task = RewTerm(
        func=mdp.SweepTaskReward,
        weight=6.0,
        params={
            "command_name": COMMAND_NAME,
            "progress_fraction": 0.7,
            "progress_rate_scale": 1.0,
            "position_tolerance": 0.025,
            "speed_tolerance": 0.02,
        },
    )
    scene_safety = RewTerm(
        func=mdp.SceneSafetyPenalty,
        weight=-1.0,
        params={
            "command_name": COMMAND_NAME,
            "non_target_speed_scale": 0.10,
        },
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness,
        weight=-0.5,
        params={"stiffness_fraction": 1.0 / 3.0},
    )
    failure = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={
            "term_keys": [
                "object_drop",
                "push_fast",
                "shelf_collision",
                "hand_velocity",
                "excessive_wrench",
            ]
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(
        func=mdp.object_drop_or_flip,
        time_out=False,
        params={
            "command_name": COMMAND_NAME,
            "minimum_height": 1.04,
            "maximum_tilt": 0.9,
        },
    )
    push_fast = DoneTerm(
        func=mdp.target_too_fast,
        time_out=False,
        params={"command_name": COMMAND_NAME, "speed_limit": 0.3},
    )
    shelf_collision = DoneTerm(
        func=mdp.shelf_collision,
        time_out=False,
        params={
            "shelf_cfg": SHELF_CFG,
            "finger_frame_cfg": FINGER_FRAME_CFG,
            "wrist_frame_cfg": WRIST_FRAME_CFG,
            "velocity_threshold": 0.1,
        },
    )
    hand_velocity = DoneTerm(
        func=mdp.arm_joint_speed,
        time_out=False,
        params={"asset_cfg": ARM_CFG, "speed_limit": 1.0},
    )
    excessive_wrench = DoneTerm(
        func=mdp.excessive_wrench,
        time_out=False,
        params={
            "ft_cfg": FT_CFG,
            "force_limit": 100.0,
            "torque_limit": 15.0,
        },
    )


@configclass
class ShelfForceEnvCfg(ManagerBasedRLEnvCfg):
    scene: ShelfForceSceneCfg = ShelfForceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
