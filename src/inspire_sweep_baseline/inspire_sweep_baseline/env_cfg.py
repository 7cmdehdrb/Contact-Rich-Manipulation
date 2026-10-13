"""Self-contained baseline configuration, derived from the v2-mass behavior."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .assets.mass_cylinder import MassCylinderCfg
from .assets.paths import SHELF_USD_PATH
from .assets.robot import ARM_JOINT_NAMES, make_robot_cfg

TARGET_Z_OFFSET = 0.09  # bottom-origin link; cylinder center is at +0.05 m.
TOOL_ROTATION_H = (0.5, -0.5, -0.5, -0.5)


def _arm():
    return SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)


def _tool_frame(targets, body="inspire_base_link", debug_vis=False):
    markers = FRAME_MARKER_CFG.copy()
    markers.markers["frame"].scale = (0.05, 0.05, 0.05)
    markers.prim_path = f"/Visuals/InspireSweepBaseline/{targets[0][0]}"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=debug_vis,
        visualizer_cfg=markers,
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}", name=name,
            offset=OffsetCfg(pos=position, rot=TOOL_ROTATION_H),
        ) for name, position in targets],
    )


@configclass
class SweepSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=SHELF_USD_PATH, mass_props=sim_utils.MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0)),
    )
    robot = make_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    object_collection = RigidObjectCollectionCfg(rigid_objects={
        "mass_cylinder": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/mass_cylinder",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.75, 0.0, 1.05)),
            spawn=MassCylinderCfg(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0, max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0, disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.001, rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
            ),
        ),
    })
    # Sensor housing center is shifted +height/2 from the original hand mount.
    # Hand base is +height: this TCP is exactly hand-local (0, 0, 0.13).
    ee_frame = _tool_frame(
        (("end_effector", (0.0, 0.0, 0.13 + robot.spawn.sensor_height / 2)),),
        body="axia80_link", debug_vis=True,
    )
    # Virtual source finger/wrist points used by shelf penalties/terminations.
    finger_frame = _tool_frame((
        ("l_finger", (0.07, 0.0, 0.13)), ("r_finger", (-0.07, 0.0, 0.13)),
    ))
    wrist_frame = _tool_frame((("wrist", (0.0, 0.0, -0.17)),))
    palm_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/inspire_palm_force_sensor",
        update_period=0.0, history_length=1, force_threshold=0.05, debug_vis=False,
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True,
        scale=0.5, use_default_offset=True,
    )
    gripper_action = mdp.HandOpennessActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Preserve the baseline observation order: 34 + palm1 + wrench6 = 41.
        joint_pos = ObsTerm(func=mdp.arm_joint_pos_rel, params={"asset_cfg": _arm()})
        joint_vel = ObsTerm(func=mdp.arm_joint_vel_rel, params={"asset_cfg": _arm()})
        actions = ObsTerm(func=mdp.last_action)
        target_obs_state = ObsTerm(
            func=mdp.object_position_in_root, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(func=mdp.object_width, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=mdp.eef_pose_in_root)
        goal_pos = ObsTerm(func=mdp.goal_position_in_root, params={"command_name": "target_goal_pos"})
        tactile = ObsTerm(func=mdp.palm_contact_bit)
        wrist_wrench = ObsTerm(func=mdp.wrist_wrench_eef)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    target_goal_pos = mdp.SweepGoalCommandCfg(asset_name="object_collection")


@configclass
class EventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(
        func=mdp.randomize_single_object_scene, mode="reset",
        params={"sweep_direction": "both", "object_radius": 0.04,
                "position_noise": 0.02, "initial_position": (-0.75, 0.0, 1.05)},
    )


@configclass
class RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.03, params={"asset_cfg": _arm()})
    shelf_collision = RewTerm(func=mdp.ShelfCollisionPenalty, weight=-0.5)
    object_collision = RewTerm(func=mdp.object_collision, weight=-0.5)
    reaching = RewTerm(func=mdp.reaching_target, weight=3.0, params={"z_offset": TARGET_Z_OFFSET})
    orientation = RewTerm(func=mdp.align_eef_y_with_shelf_z, weight=2.0)
    sweeping_object = RewTerm(
        func=mdp.pushing_target, weight=6.0,
        params={"command_name": "target_goal_pos", "z_offset": TARGET_Z_OFFSET},
    )
    homing_after_sweep = RewTerm(
        func=mdp.homing_reward, weight=9.0,
        params={"command_name": "target_goal_pos", "asset_cfg": _arm()},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.object_drop, params={"height_condition": 1.04, "rotation_condition": 0.9})
    push_fast = DoneTerm(func=mdp.push_fast, params={"speed_condition": 0.3})
    shelf_collision = DoneTerm(func=mdp.shelf_collision, params={"threshold": 0.1})
    # Historical term name retained in logs; it restricts the six ARM joints.
    hand_velocity = DoneTerm(func=mdp.arm_velocity, params={"asset_cfg": _arm(), "threshold": 1.0})


@configclass
class InspireSweepEnvCfg(ManagerBasedRLEnvCfg):
    scene: SweepSceneCfg = SweepSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(3.5, 3.5, 3.5), lookat=(0.0, 0.0, 0.3))

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
