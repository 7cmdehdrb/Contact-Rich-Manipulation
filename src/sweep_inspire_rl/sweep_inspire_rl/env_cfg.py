"""Sweep-Policy shelf task with the Axia80/Inspire robot configuration.

The reference shelf USD, object catalog, reset position grid, physics timing,
command, reward weights, and termination limits are preserved. Only hand,
single-object, relative-frame, and rightward-sweep adaptations live here.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hand_manipulation_rl.assets.robot import (
    ARM_JOINT_NAMES,
    HAND_BASE_BODY_NAME,
    HAND_JOINT_NAMES,
    PALM_SENSOR_BODY_NAMES,
    FT_SENSOR_BODY_NAME,
    make_robot_cfg,
)
from sweeping_policy import mdp as sweep_mdp
from sweeping_policy.shelf_sweep_random_env_cfg import ShelfSweepRandomEnvCfg
from sweeping_policy.src.assets import ENVIRONMENT_YAML_PATH, object_usd_path
from sweeping_policy.src.shelf_utils import load_and_reshape_pose
import yaml

from .mdp.actions import NearlyOpenHandSynergyActionCfg, RightPalmOscActionCfg
from .mdp.events import initialize_right_palm_at_target, randomize_single_target
from .mdp.observations import actual_hand_synergy, palm_tactile_bits, wrist_wrench_c
from .mdp import rewards

C_OFFSET_H = (0.0, 0.05, 0.10)
RIGHT_PALM_QUAT_W = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
POLICY_OBSERVATION_DIM = 71


@configclass
class ActionsCfg:
    arm_action = RightPalmOscActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=HAND_BASE_BODY_NAME,
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=C_OFFSET_H),
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            # The source's gravity-off setting sagged 36 mm with this hand
            # and hit the arm velocity limit before reaching the object.
            gravity_compensation=True,
            motion_stiffness_task=(200.0,) * 6,
            motion_damping_ratio_task=(1.0,) * 6,
            nullspace_control="none",
        ),
        nullspace_joint_pos_target="none",
        position_scale=1.0,
        orientation_scale=1.0,
    )
    hand_action = NearlyOpenHandSynergyActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Preserve reference term order, replacing the gripper joint slice with
        # named arm+hand joints and adding the actual synergy and sensors.
        joint_pos = ObsTerm(
            func=lab_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(ARM_JOINT_NAMES + HAND_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=lab_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
                )
            },
        )
        actions = ObsTerm(func=lab_mdp.last_action)
        target_obs_state = ObsTerm(
            func=sweep_mdp.MA_object_position_in_EEF,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(
            func=sweep_mdp.MA_object_width, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee_pose = ObsTerm(func=sweep_mdp.ee_pos_r)
        goal_pos = ObsTerm(
            func=sweep_mdp.MA_target_goal_command_in_EEF,
            params={"command_name": "target_goal_pos"},
        )
        hand_synergy = ObsTerm(func=actual_hand_synergy)
        tactile = ObsTerm(func=palm_tactile_bits)
        wrist_wrench = ObsTerm(func=wrist_wrench_c, params={"c_offset_h": C_OFFSET_H})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_all = EventTerm(func=lab_mdp.reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(func=randomize_single_target, mode="reset", params={})
    initialize_palm = EventTerm(
        func=initialize_right_palm_at_target,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_collection_cfg": SceneEntityCfg("object_collection"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "eef_body_name": HAND_BASE_BODY_NAME,
            "eef_offset": C_OFFSET_H,
            # C lies 19.6 mm toward the wrist from the palm pad in world X.
            # Align the actual pad center with the object's shelf depth.
            "reaching_x_offset": 0.0196,
            "reaching_z_offset": 0.12,
            "side_clearance": 0.04,
            "position_noise": 0.002,
            "max_iterations": 160,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.005,
            "orientation_tolerance": math.radians(1.0),
            "joint_seed_offsets": (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.2, -0.2, 0.2, 0.0, 0.0, -0.8),
                (-0.2, 0.2, -0.2, 0.0, 0.0, 0.8),
                (0.4, -0.3, 0.3, 0.2, -0.1, -1.6),
                (-0.4, 0.3, -0.3, -0.2, 0.1, 1.6),
            ),
        },
    )


@configclass
class InspireShelfSweepEnvCfg(ShelfSweepRandomEnvCfg):
    """One source-catalog object; palm faces right throughout the episode."""

    object_name: str = "cup_1"
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = make_robot_cfg()
        catalog = self._load_object_catalog()
        if self.object_name not in catalog["objects"]:
            raise ValueError(
                f"Unknown object_name {self.object_name!r}; choose {tuple(catalog['objects'])}"
            )
        target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(catalog["pose"][self.object_name][:3]),
                rot=tuple(catalog["pose"][self.object_name][3:]),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=object_usd_path(catalog["objects"][self.object_name]),
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
        self.scene.object_collection = RigidObjectCollectionCfg(
            rigid_objects={"target": target}
        )
        self.scene.ee_frame = self._frame(
            "control_point", HAND_BASE_BODY_NAME, C_OFFSET_H
        )
        self.scene.finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}", name=name
                )
                for name in (
                    "inspire_middle_force_sensor_3",
                    "inspire_little_force_sensor_3",
                )
            ],
        )
        self.scene.wrist_frame = self._frame("axia80", FT_SENSOR_BODY_NAME)
        self.scene.palm_tactile = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")",
            update_period=0.0,
            history_length=1,
            force_threshold=0.05,
            debug_vis=False,
        )
        self.commands.target_goal_pos.asset_name = "object_collection"
        self.commands.target_goal_pos.asset_dict = {"target": target}
        self.commands.target_goal_pos.object_id_dict_rev = {"0": "target"}
        self.events.object_spawn.params = {
            "pose_array": self._reset_pose_array(catalog),
            "object_width": float(catalog["width"][self.object_name]),
        }
        self.rewards.joint_vel.func = lab_mdp.joint_vel_l2
        self.rewards.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
            )
        }
        self.rewards.reaching.func = rewards.hand_reaching
        self.rewards.orientation.func = rewards.palm_alignment
        self.rewards.sweeping_object.func = rewards.pushing_target
        self.rewards.sweeping_object.params["eef_distance_threshold"] = 0.09
        self.rewards.homing_after_sweep = None
        self.terminations.object_drop.params = {
            "height_condition": 1.04,
            "rotation_condition": 0.9,
        }
        self.terminations.push_fast.params["speed_condition"] = 0.3
        self.terminations.hand_velocity.func = rewards.hand_velocity_limit
        # Sweep's shelf geometry, timing, capacities, rewards/curriculum, and
        # thresholds otherwise stay in the source configuration.

    @staticmethod
    def _reset_pose_array(catalog):
        return load_and_reshape_pose(catalog["pose"])

    @staticmethod
    def _load_object_catalog():
        with open(ENVIRONMENT_YAML_PATH, encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def _frame(name, body, offset=(0.0, 0.0, 0.0)):
        return FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                    name=name,
                    offset=OffsetCfg(pos=offset),
                )
            ],
        )


@configclass
class InspireShelfSweepEnvCfg_PLAY(InspireShelfSweepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class InspireShelfSweepV1EnvCfg(InspireShelfSweepEnvCfg):
    """Reach through the object's XY center; use a 4 cm planar EEF gate."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching.func = rewards.hand_reaching_object_center
        self.rewards.reaching.params = {"z_offset": 0.075}
        self.rewards.sweeping_object.params["eef_distance_threshold"] = 0.04
        self.rewards.sweeping_object.params["eef_distance_xy_only"] = True
        self.rewards.sweeping_object.params["pushing_z_offset"] = 0.075


@configclass
class InspireShelfSweepV1EnvCfg_PLAY(InspireShelfSweepV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class InspireShelfSweepV2EnvCfg(InspireShelfSweepV1EnvCfg):
    """V1 with episode-fixed height and a sweeping height penalty."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching.func = rewards.hand_reaching_fixed_height
        self.rewards.reaching.params = {"z_offset": 0.075, "command_name": "target_goal_pos"}
        self.rewards.sweeping_object.params["height_reference_initial"] = True
        self.rewards.sweeping_height = RewTerm(
            func=rewards.sweeping_height_error,
            weight=-1.0,
            params={
                "z_offset": 0.075,
                "height_scale": 0.015,
                "eef_distance_threshold": 0.04,
                "command_name": "target_goal_pos",
            },
        )


@configclass
class InspireShelfSweepV2EnvCfg_PLAY(InspireShelfSweepV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False


@configclass
class InspireShelfSweepV3EnvCfg(InspireShelfSweepV1EnvCfg):
    """Unmodified V1 rewards with a single low-COM, weighted cylinder."""

    object_name: str = "weighted_cylinder"

    def __post_init__(self):
        if self.object_name != "weighted_cylinder":
            raise ValueError("V3 uses only object_name='weighted_cylinder'")
        super().__post_init__()
        # Preserve the mass, COM and inertia authored in the local cylinder USD.
        self.scene.object_collection.rigid_objects["target"].spawn.mass_props = None

    @staticmethod
    def _reset_pose_array(catalog):
        return load_and_reshape_pose({
            name: pose for name, pose in catalog["pose"].items() if name != "weighted_cylinder"
        })

    @staticmethod
    def _load_object_catalog():
        catalog = InspireShelfSweepEnvCfg._load_object_catalog()
        asset = Path(__file__).resolve().parent / "assets" / "weighted_cylinder.usda"
        catalog["objects"]["weighted_cylinder"] = str(asset)
        catalog["pose"]["weighted_cylinder"] = list(catalog["pose"]["cup_1"])
        # The source's width parameter is used as a lateral standoff; use the
        # cylinder radius so the source XY gate overlaps its contact surface.
        catalog["width"]["weighted_cylinder"] = 0.04
        return catalog


@configclass
class InspireShelfSweepV3EnvCfg_PLAY(InspireShelfSweepV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
