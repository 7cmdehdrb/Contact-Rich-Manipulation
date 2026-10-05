"""Table-supported Cube contact specialization of the free-space reaching task."""

from dataclasses import fields
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...assets.contact_robot import make_contact_robot_cfg
from ...assets.robot import PALM_SENSOR_BODY_NAMES, ROBOT_CONTACT_BODY_NAMES
from ...env_cfg import CommandsCfg, EventsCfg, ReachTaskCfg, RewardsCfg, TerminationsCfg
from ...mdp.contact_commands import CubeInitialPositionCommandCfg
from ...mdp.contact_events import ContactSafePoseReset, reset_cube_on_table
from ...mdp.contact_rewards import palm_cube_contact_reward
from ...mdp.contact_terminations import contact_time_out, table_contact_failure
from ...mdp.rewards import ExecutedActionRate
from .reach_env_cfg import UR5eInspireReachEnvCfg, UR5eInspireReachSceneCfg


@configclass
class ContactTaskCfg(ReachTaskCfg):
    tactile_threshold_n: float = 0.01
    contact_threshold_n: float = 0.01
    table_size: tuple[float, float, float] = (0.36, 1.00, 0.04)
    table_center: tuple[float, float, float] = (-0.75, 0.0, 1.03)
    cube_size: float = 0.06
    cube_mass: float = 1.0
    object_xy_offset_low: tuple[float, float] = (-0.07, -0.22)
    object_xy_offset_high: tuple[float, float] = (0.12, 0.22)
    reset_position_offset_task: tuple[float, float, float] = (0.14, 0.0, 0.10)
    reset_position_jitter_task: tuple[float, float, float] = (0.004, 0.004, 0.003)
    reset_c_y_range: tuple[float, float] = (-0.25, 0.25)
    initial_hand_open_range: tuple[float, float] = (0.90, 0.98)
    reset_joint_seed_offsets: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.20, -0.12, 0.12, 0.0, 0.0, -0.15),
        (-0.20, -0.12, 0.12, 0.0, 0.0, 0.15),
    )
    reset_max_iterations: int = 80
    reset_damping: float = 0.045
    reset_step_size: float = 0.65
    reset_position_error_step_m: float = 0.06
    reset_orientation_error_step_rad: float = 0.25
    reset_joint_delta_limit_rad: float = 0.18
    reset_position_tolerance_m: float = 0.003
    reset_orientation_tolerance_rad: float = 0.05
    reset_joint_limit_margin_rad: float = 0.035
    reset_wrist_3_range_rad: tuple[float, float] = (-2.8, 2.8)
    reset_clearance_margin_m: float = 0.004

    @property
    def cube_center_height_m(self) -> float:
        return self.table_center[2] + 0.5 * (self.table_size[2] + self.cube_size)

    @property
    def object_xy_range_low(self) -> tuple[float, float]:
        return tuple(self.table_center[axis] + self.object_xy_offset_low[axis] for axis in range(2))

    @property
    def object_xy_range_high(self) -> tuple[float, float]:
        return tuple(self.table_center[axis] + self.object_xy_offset_high[axis] for axis in range(2))

    def validate(self):
        # The parent assumes every instance attribute is a scalar. Restrict
        # those checks to its fields, then validate the new geometry explicitly.
        scalar_names = [item.name for item in fields(ReachTaskCfg)] + [
            "contact_threshold_n", "cube_size", "cube_mass", "reset_damping", "reset_step_size",
            "reset_position_error_step_m", "reset_orientation_error_step_rad", "reset_joint_delta_limit_rad",
            "reset_position_tolerance_m", "reset_orientation_tolerance_rad", "reset_joint_limit_margin_rad",
            "reset_clearance_margin_m",
        ]
        for name in scalar_names:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        vectors = (
            ("table_size", 3), ("table_center", 3), ("object_xy_offset_low", 2),
            ("object_xy_offset_high", 2), ("object_xy_range_low", 2),
            ("object_xy_range_high", 2), ("reset_position_offset_task", 3),
            ("reset_position_jitter_task", 3), ("reset_c_y_range", 2),
            ("initial_hand_open_range", 2), ("reset_wrist_3_range_rad", 2),
        )
        for name, size in vectors:
            values = getattr(self, name)
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {size} finite values")
        if any(value <= 0.0 for value in self.table_size):
            raise ValueError("table_size must be positive")
        if any(value < 0.0 for value in self.reset_position_jitter_task):
            raise ValueError("reset_position_jitter_task must be non-negative")
        if self.reset_position_offset_task[0] <= self.cube_size / 2 or self.reset_position_offset_task[2] <= 0:
            raise ValueError("Reset backoff and height must preserve a positive separation")
        if not 0.0 <= self.initial_hand_open_range[0] <= self.initial_hand_open_range[1] <= 1.0:
            raise ValueError("initial_hand_open_range must be within [0, 1]")
        for name in ("reset_c_y_range", "reset_wrist_3_range_rad"):
            low, high = getattr(self, name)
            if low >= high:
                raise ValueError(f"{name} must be increasing")
        if not isinstance(self.reset_max_iterations, int) or self.reset_max_iterations <= 0:
            raise ValueError("reset_max_iterations must be a positive integer")
        if not self.reset_joint_seed_offsets or any(
            len(seed) != 6 or not all(math.isfinite(value) for value in seed)
            for seed in self.reset_joint_seed_offsets
        ):
            raise ValueError("reset_joint_seed_offsets must contain finite six-joint offsets")
        for axis in range(2):
            low, high = self.object_xy_range_low[axis], self.object_xy_range_high[axis]
            surface_low = self.table_center[axis] - self.table_size[axis] / 2 + self.cube_size / 2
            surface_high = self.table_center[axis] + self.table_size[axis] / 2 - self.cube_size / 2
            if low > high or low < surface_low - 1e-8 or high > surface_high + 1e-8:
                raise ValueError("Cube sampling range must fit on the table")


_TASK = ContactTaskCfg()


def _robot_filter_paths(body_names):
    return ["{ENV_REGEX_NS}/Robot/" + name for name in body_names]


@configclass
class UR5eInspireContactSceneCfg(UR5eInspireReachSceneCfg):
    robot = make_contact_robot_cfg()
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=_TASK.table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
                solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.25,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85, dynamic_friction=0.65, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.34, 0.25)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TASK.table_center),
    )
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=(_TASK.cube_size,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_TASK.cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65, dynamic_friction=0.45, restitution=0.0,
                friction_combine_mode="average", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.43, 0.90)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(
            (_TASK.object_xy_range_low[0] + _TASK.object_xy_range_high[0]) / 2,
            (_TASK.object_xy_range_low[1] + _TASK.object_xy_range_high[1]) / 2,
            _TASK.cube_center_height_m,
        )),
    )
    table_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Table", update_period=0.0, history_length=2,
        filter_prim_paths_expr=_robot_filter_paths(ROBOT_CONTACT_BODY_NAMES),
        max_contact_data_count_per_prim=256, debug_vis=False,
    )
    cube_palm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject", update_period=0.0, history_length=0,
        filter_prim_paths_expr=_robot_filter_paths(PALM_SENSOR_BODY_NAMES),
        max_contact_data_count_per_prim=128, debug_vis=False,
    )


@configclass
class ContactCommandsCfg(CommandsCfg):
    target_position = CubeInitialPositionCommandCfg(debug_vis=True)


@configclass
class ContactEventsCfg(EventsCfg):
    reset_cube = EventTermCfg(func=reset_cube_on_table, mode="reset")
    safe_hand = EventTermCfg(func=ContactSafePoseReset, mode="reset")


@configclass
class ContactRewardsCfg(RewardsCfg):
    action_rate = RewardTermCfg(func=ExecutedActionRate, weight=0.001)
    palm_cube_contact = RewardTermCfg(func=palm_cube_contact_reward, weight=0.02)


@configclass
class ContactTerminationsCfg(TerminationsCfg):
    time_out = TerminationTermCfg(func=contact_time_out, time_out=True)
    table_contact = TerminationTermCfg(func=table_contact_failure)


@configclass
class UR5eInspireContactEnvCfg(UR5eInspireReachEnvCfg):
    task: ContactTaskCfg = ContactTaskCfg()
    scene: UR5eInspireContactSceneCfg = UR5eInspireContactSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, lazy_sensor_update=False,
    )
    commands: ContactCommandsCfg = ContactCommandsCfg()
    events: ContactEventsCfg = ContactEventsCfg()
    rewards: ContactRewardsCfg = ContactRewardsCfg()
    terminations: ContactTerminationsCfg = ContactTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.table.spawn.size = self.task.table_size
        self.scene.table.init_state.pos = self.task.table_center
        self.scene.target_object.spawn.size = (self.task.cube_size,) * 3
        self.scene.target_object.spawn.mass_props.mass = self.task.cube_mass
        self.scene.target_object.init_state.pos = (
            (self.task.object_xy_range_low[0] + self.task.object_xy_range_high[0]) / 2,
            (self.task.object_xy_range_low[1] + self.task.object_xy_range_high[1]) / 2,
            self.task.cube_center_height_m,
        )
        self.scene.table_contacts.history_length = self.decimation
        self.scene.lazy_sensor_update = False
        self.viewer.lookat = (*self.task.table_center[:2], self.task.cube_center_height_m)
