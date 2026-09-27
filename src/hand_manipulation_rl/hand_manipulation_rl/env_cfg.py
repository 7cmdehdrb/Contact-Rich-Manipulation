"""Configuration for the independent blind-sweeping feasibility task.

Every numerical value in this file is an explicit bring-up proposal. None of
the geometry, thresholds, gains, or wrench parameters are measured values;
they are centralized here so a scene probe can replace them without changing
the MDP implementation.
"""

from __future__ import annotations

import math

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .assets.robot import (
    ARM_JOINT_NAMES,
    HAND_BASE_BODY_NAME,
    HAND_CONTACT_BODY_NAMES,
    PALM_SENSOR_BODY_NAMES,
    ROBOT_CONTACT_BODY_NAMES,
    make_robot_cfg,
)
from .mdp.actions import CurrentFrameOscActionCfg, InspireHandSynergyActionCfg
from .mdp.events import StableOffsetPoseReset, reset_episode_scene
from .mdp.observations import policy_observation
from .mdp.rewards import (
    actual_contact_normal_alignment,
    board_contact_risk_penalty,
    failure_terminal_penalty_rate,
    normalized_action_change,
    object_goal_reward,
    object_height_risk_penalty,
    selected_tactile_contact,
    success_terminal_bonus_rate,
    tilt_risk_penalty,
    time_cost_rate,
)
from .mdp.terminations import (
    board_contact_failure,
    episode_timeout,
    height_limit_failure,
    invalid_reset_contact,
    object_out_of_bounds,
    task_success,
    topple_failure,
)
from .sensors import DORSAL_PARENT_BODY_NAMES


# PROPOSED bring-up geometry. These values live at module scope as well as in
# BlindSweepTaskCfg because scene entities are constructed before an environment
# instance exists.
# Thin-Cuboid equivalent of Sweep-Policy's active shelf board.  Axes are
# (depth-X, width-Y, thickness-Z); the width is deliberately much larger than
# the depth and the top surface is at the reference z=1.05 m.
BOARD_SIZE_M = (0.36, 1.00, 0.04)
BOARD_CENTER_S_M = (-0.70, 0.00, 1.03)
CUBE_SIZE_M = 0.060
CUBE_INITIAL_CENTER_S_M = (
    -0.675,
    0.00,
    BOARD_CENTER_S_M[2] + 0.5 * BOARD_SIZE_M[2] + 0.5 * CUBE_SIZE_M,
)
OBSTACLE_CONTACT_OFFSET_M = 0.002

# H is inspire_base_link. +Y is the palmar outward normal and +Z runs toward
# the fingers in the package-local hand model. C is a massless computed frame.
C_OFFSET_H_M = (0.0, 0.050, 0.100)
C_QUAT_H_WXYZ = (1.0, 0.0, 0.0, 0.0)


@configclass
class BlindSweepTaskCfg:
    """All task-specific values, including deliberately unvalidated proposals."""

    # Geometry and episode command distribution.
    board_size: tuple[float, float, float] = BOARD_SIZE_M
    board_center: tuple[float, float, float] = BOARD_CENTER_S_M
    board_edge_margin: float = 0.060
    cube_size: float = CUBE_SIZE_M
    # (shelf depth-X, shelf width-Y), matching the reference work envelope.
    object_xy_range_low: tuple[float, float] = (-0.77, -0.22)
    object_xy_range_high: tuple[float, float] = (-0.58, 0.22)
    initial_cube_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    command_distance_range: tuple[float, float] = (0.18, 0.18)
    command_sample_attempts: int = 128
    palm_mode_probability: float = 0.50

    # Fixed virtual frame and reset variation.
    c_offset_h: tuple[float, float, float] = C_OFFSET_H_M
    c_quat_h: tuple[float, float, float, float] = C_QUAT_H_WXYZ
    # Includes the cube half-extent, the complete sampled open-hand collision
    # envelope, and a positive no-contact margin.  The vertical offset keeps
    # the hand's lower lateral edge above the board while its face overlaps
    # the cube height.
    c_standoff: float = 0.140
    c_vertical_offset: float = 0.075
    c_longitudinal_variation: tuple[float, float] = (-0.010, 0.010)
    c_lateral_variation: tuple[float, float] = (-0.015, 0.015)
    c_height_variation: tuple[float, float] = (-0.006, 0.006)
    c_roll_variation: tuple[float, float] = (-0.08, 0.08)
    c_pitch_variation: tuple[float, float] = (-0.08, 0.08)
    c_yaw_variation: tuple[float, float] = (-0.10, 0.10)
    initial_hand_open_range_low: tuple[float, float] = (0.72, 0.72)
    initial_hand_open_range_high: tuple[float, float] = (0.95, 0.95)

    # Full-pose reset IK.
    ik_max_iterations: int = 96
    ik_damping: float = 0.045
    ik_step_size: float = 0.65
    ik_position_tolerance_m: float = 0.003
    ik_orientation_tolerance_rad: float = 0.050
    ik_singular_value_min: float = 0.008
    ik_joint_limit_margin_rad: float = 0.035
    ik_wrist_3_range_rad: tuple[float, float] = (-2.80, 2.80)
    ik_wrist_3_cost_weight: float = 2.0
    reset_workspace_reach_range_m: tuple[float, float] = (0.18, 0.95)
    reset_workspace_height_range_m: tuple[float, float] = (0.12, 0.82)
    ik_max_spec_resamples: int = 64
    ik_joint_seed_offsets: tuple[tuple[float, ...], ...] = (
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (0.30, -0.20, 0.20, 0.00, 0.00, -0.25),
        (-0.30, -0.20, 0.20, 0.00, 0.00, 0.25),
        (0.55, 0.15, -0.25, 0.15, 0.00, -0.35),
        (-0.55, 0.15, -0.25, -0.15, 0.00, 0.35),
    )

    # Active stable offset reset. Coordinates are (backoff opposite the sweep,
    # shelf tangent, world-up height) relative to the Cube center. The narrow
    # jitter and nearly-open Hand deliberately favor collision-free starts.
    stable_reset_position_offset_task: tuple[float, float, float] = (0.140, 0.0, 0.100)
    stable_reset_position_jitter_task: tuple[float, float, float] = (0.004, 0.004, 0.003)
    stable_reset_c_y_range: tuple[float, float] = (-0.25, 0.25)
    stable_reset_orientation_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stable_reset_orientation_jitter_rpy: tuple[float, float, float] = (
        math.radians(1.0),
        math.radians(1.0),
        math.radians(2.0),
    )
    stable_reset_hand_open_range: tuple[float, float] = (0.90, 0.98)
    stable_reset_max_pose_attempts: int = 3
    stable_reset_max_iterations: int = 80
    stable_reset_position_error_step_m: float = 0.060
    stable_reset_orientation_error_step_rad: float = 0.250
    stable_reset_joint_seed_offsets: tuple[tuple[float, ...], ...] = (
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (0.20, -0.12, 0.12, 0.00, 0.00, -0.15),
        (-0.20, -0.12, 0.12, 0.00, 0.00, 0.15),
        (0.40, 0.10, -0.20, 0.10, 0.00, -0.25),
        (-0.40, 0.10, -0.20, -0.10, 0.00, 0.25),
    )

    # Current-C-frame action and fixed OSC gains.
    arm_translation_action_scale_m: tuple[float, float, float] = (0.012, 0.012, 0.008)
    arm_rotation_action_scale_rad: tuple[float, float, float] = (0.060, 0.060, 0.060)
    osc_motion_stiffness: tuple[float, float, float, float, float, float] = (
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    )
    osc_motion_damping_ratio: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    osc_effort_limit_scale: float = 0.90
    hand_max_synergy_rate_per_s: tuple[float, float] = (1.5, 1.5)
    hand_max_joint_target_rate_rad_s: tuple[float, ...] = (1.0,) * 12

    # Observation scales and sensor thresholds.
    arm_velocity_observation_scale: float = 3.14
    position_observation_scale_m: float = 0.20
    rotation_observation_scale_rad: float = math.pi
    distance_observation_scale_m: float = 0.20
    wrench_force_observation_scale_n: float = 40.0
    wrench_moment_observation_scale_nm: float = 4.0
    tactile_threshold_n: float = 0.05
    dorsal_tactile_threshold_n: float = 0.05

    # Physical-contact tolerance used by reset/smoke certification.  F/T
    # readings themselves are never gravity- or tare-cancelled.
    reset_contact_tolerance_n: float = 0.05
    reset_clearance_margin_m: float = 0.004
    surface_alignment_min_dot: float = 0.90

    # Reward terms. RewardManager performs the sole policy-dt integration.
    goal_reward_sigma_m: float = 0.050
    tactile_contact_beta: float = 0.65
    arm_action_change_coefficient: float = 1.0 / 6.0
    hand_action_change_coefficient: float = 1.0 / 2.0
    time_reference_s: float = 4.0
    success_terminal_bonus: float = 2.0
    failure_terminal_penalty: float = -2.0

    # Soft/hard risk bands and success.
    success_distance_m: float = 0.010
    tilt_soft_rad: float = math.radians(12.0)
    tilt_hard_rad: float = math.radians(30.0)
    board_force_soft_n: float = 2.0
    board_force_hard_n: float = 10.0
    height_soft_m: float = 0.012
    height_hard_m: float = 0.035
    enable_out_of_bounds_termination: bool = True
    out_of_bounds_margin_m: float = 0.030

    def validate(self) -> None:
        """Fail early on contradictions that would silently change the task."""

        if len(self.board_size) != 3 or any(
            value <= 0.0 or not math.isfinite(value) for value in self.board_size
        ):
            raise ValueError("board_size must contain three positive values")
        if len(self.board_center) != 3 or any(not math.isfinite(value) for value in self.board_center):
            raise ValueError("board_center must contain three finite values")
        if self.board_size[1] <= self.board_size[0]:
            raise ValueError("shelf width (Y) must be larger than shelf depth (X)")
        if (
            self.cube_size <= 0.0
            or not math.isfinite(self.cube_size)
            or not math.isfinite(self.board_edge_margin)
            or self.board_edge_margin < 0.5 * self.cube_size
        ):
            raise ValueError("board_edge_margin must include at least the cube half extent")
        if len(self.object_xy_range_low) != 2 or len(self.object_xy_range_high) != 2:
            raise ValueError("object XY sampling ranges must contain two values")
        if any(
            not math.isfinite(value)
            for value in (*self.object_xy_range_low, *self.object_xy_range_high)
        ):
            raise ValueError("object XY sampling ranges must be finite")
        if any(low >= high for low, high in zip(self.object_xy_range_low, self.object_xy_range_high)):
            raise ValueError("object XY sampling ranges must be strictly increasing")
        usable_lower = tuple(
            self.board_center[axis] - 0.5 * self.board_size[axis] + self.board_edge_margin
            for axis in range(2)
        )
        usable_upper = tuple(
            self.board_center[axis] + 0.5 * self.board_size[axis] - self.board_edge_margin
            for axis in range(2)
        )
        if any(lower >= upper for lower, upper in zip(usable_lower, usable_upper)):
            raise ValueError("board_edge_margin leaves no usable planar board area")
        boundary_tolerance = 1.0e-9
        if any(
            sample_low < board_low - boundary_tolerance
            or sample_high > board_high + boundary_tolerance
            for sample_low, sample_high, board_low, board_high in zip(
                self.object_xy_range_low,
                self.object_xy_range_high,
                usable_lower,
                usable_upper,
            )
        ):
            raise ValueError("object XY sampling ranges must stay inside the usable board area")
        if not 0.0 <= self.palm_mode_probability <= 1.0:
            raise ValueError("palm_mode_probability must lie in [0, 1]")
        if self.command_distance_range[0] <= 0.0 or self.command_distance_range[0] > self.command_distance_range[1]:
            raise ValueError("command distance range must be finite, positive, and ordered")
        if self.command_sample_attempts <= 0 or self.ik_max_spec_resamples <= 0:
            raise ValueError("sampling retry counts must be positive")
        for name, values in (
            ("stable reset position offset", self.stable_reset_position_offset_task),
            ("stable reset position jitter", self.stable_reset_position_jitter_task),
            ("stable reset orientation offset", self.stable_reset_orientation_offset_rpy),
            ("stable reset orientation jitter", self.stable_reset_orientation_jitter_rpy),
        ):
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
        if (
            self.stable_reset_position_offset_task[0] <= 0.5 * self.cube_size
            or self.stable_reset_position_offset_task[2] <= 0.5 * self.cube_size
            or any(value < 0.0 for value in self.stable_reset_position_jitter_task)
            or any(value < 0.0 for value in self.stable_reset_orientation_jitter_rpy)
        ):
            raise ValueError("stable reset offsets/jitter must preserve positive backoff and height")
        if (
            len(self.stable_reset_hand_open_range) != 2
            or not 0.0 < self.stable_reset_hand_open_range[0]
            <= self.stable_reset_hand_open_range[1]
            <= 1.0
        ):
            raise ValueError("stable reset hand-open range must lie in (0, 1]")
        if self.stable_reset_max_pose_attempts <= 0 or self.stable_reset_max_iterations <= 0:
            raise ValueError("stable reset retry and iteration counts must be positive")
        if (
            self.stable_reset_position_error_step_m <= 0.0
            or self.stable_reset_orientation_error_step_rad <= 0.0
        ):
            raise ValueError("stable reset Cartesian error steps must be positive")
        if (
            len(self.stable_reset_c_y_range) != 2
            or not all(math.isfinite(value) for value in self.stable_reset_c_y_range)
            or self.stable_reset_c_y_range[0] >= self.stable_reset_c_y_range[1]
        ):
            raise ValueError("stable reset C y range must contain two finite increasing values")
        if not self.stable_reset_joint_seed_offsets or any(
            len(seed) != 6 or not all(math.isfinite(value) for value in seed)
            for seed in self.stable_reset_joint_seed_offsets
        ):
            raise ValueError("stable reset joint seeds must contain finite six-joint offsets")
        if len(self.c_offset_h) != 3 or len(self.c_quat_h) != 4:
            raise ValueError("the fixed H-to-C transform must be 3D position plus quaternion")
        if self.c_standoff <= 0.5 * self.cube_size or self.c_vertical_offset < 0.0:
            raise ValueError("C standoff/vertical offset must preserve a positive collision margin")
        if len(self.hand_max_joint_target_rate_rad_s) != 12:
            raise ValueError("the hand joint-rate configuration must contain twelve values")
        if self.reset_contact_tolerance_n <= 0.0:
            raise ValueError("reset no-contact tolerance must be positive")
        if self.reset_clearance_margin_m < 0.0:
            raise ValueError("reset clearance margin must be non-negative")
        if self.ik_wrist_3_cost_weight < 0.0 or not math.isfinite(self.ik_wrist_3_cost_weight):
            raise ValueError("ik_wrist_3_cost_weight must be finite and non-negative")
        for name, limits in (
            ("reset workspace reach", self.reset_workspace_reach_range_m),
            ("reset workspace height", self.reset_workspace_height_range_m),
        ):
            if len(limits) != 2 or not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]:
                raise ValueError(f"{name} range must contain two finite, increasing values")
        if not 0.0 < self.surface_alignment_min_dot <= 1.0:
            raise ValueError("surface_alignment_min_dot must lie in (0, 1]")
        for name, soft, hard in (
            ("tilt", self.tilt_soft_rad, self.tilt_hard_rad),
            ("board force", self.board_force_soft_n, self.board_force_hard_n),
            ("height", self.height_soft_m, self.height_hard_m),
        ):
            if not 0.0 <= soft < hard:
                raise ValueError(f"{name} thresholds must satisfy 0 <= soft < hard")
        positive_values = {
            "success_distance_m": self.success_distance_m,
            "goal_reward_sigma_m": self.goal_reward_sigma_m,
            "time_reference_s": self.time_reference_s,
            "position_observation_scale_m": self.position_observation_scale_m,
            "rotation_observation_scale_rad": self.rotation_observation_scale_rad,
            "distance_observation_scale_m": self.distance_observation_scale_m,
            "wrench_force_observation_scale_n": self.wrench_force_observation_scale_n,
            "wrench_moment_observation_scale_nm": self.wrench_moment_observation_scale_nm,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0.0 or not math.isfinite(value)]
        if invalid:
            raise ValueError(f"task values must be finite and positive: {invalid}")


_TASK_DEFAULTS = BlindSweepTaskCfg()
_DORSAL_BODY_REGEX = "(" + "|".join(DORSAL_PARENT_BODY_NAMES) + ")"
_PALM_BODY_REGEX = "(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")"
_ROBOT_BODY_REGEX = "(" + "|".join(ROBOT_CONTACT_BODY_NAMES) + ")"


def _robot_filter_paths(body_names: tuple[str, ...]) -> list[str]:
    return [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in body_names]


@configclass
class BlindSweepSceneCfg(InteractiveSceneCfg):
    """One fixed board, one dynamic cube, and one assembled robot per environment."""

    board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        spawn=sim_utils.CuboidCfg(
            size=BOARD_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.25,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=OBSTACLE_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85,
                dynamic_friction=0.65,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.34, 0.25)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOARD_CENTER_S_M),
    )

    robot = make_robot_cfg()

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=OBSTACLE_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.45,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.43, 0.90)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUBE_INITIAL_CENTER_S_M,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Actor tactile sees all external contacts. Filtering it to TargetObject
    # would leak target identity and make the simulated sensor selective.
    palm_tactile = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_PALM_BODY_REGEX}",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[],
        max_contact_data_count_per_prim=16,
        debug_vis=False,
    )
    dorsal_tactile = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_DORSAL_BODY_REGEX}",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[],
        max_contact_data_count_per_prim=16,
        debug_vis=False,
    )

    # Reverse one-to-many sensors keep exactly one observed rigid body and
    # expose a filter dimension without vector cancellation.
    board_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=_robot_filter_paths(ROBOT_CONTACT_BODY_NAMES),
        max_contact_data_count_per_prim=128,
        debug_vis=False,
    )
    target_hand_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=_robot_filter_paths(HAND_CONTACT_BODY_NAMES),
        max_contact_data_count_per_prim=128,
        debug_vis=False,
    )
    target_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=_robot_filter_paths(ROBOT_CONTACT_BODY_NAMES),
        max_contact_data_count_per_prim=128,
        debug_vis=False,
    )
    # Unfiltered per-body forces are used only for reset/external-contact
    # validation.  The source hand requires adjacent self-collision filtering,
    # so articulation self-collision is disabled in the package-local spawner.
    robot_contacts = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_ROBOT_BODY_REGEX}",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[],
        max_contact_data_count_per_prim=32,
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    arm_action = CurrentFrameOscActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=HAND_BASE_BODY_NAME,
        body_offset_pos=C_OFFSET_H_M,
        body_offset_quat=C_QUAT_H_WXYZ,
        translation_scale=(0.012, 0.012, 0.008),
        rotation_scale=(0.060, 0.060, 0.060),
        motion_stiffness=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        gravity_compensation=False,
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        effort_limit_scale=0.90,
    )
    hand_action = InspireHandSynergyActionCfg(
        asset_name="robot",
        max_synergy_rate=(1.5, 1.5),
        max_joint_target_rate=(1.0,) * 12,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        vector = ObsTerm(func=policy_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Exactly two reset events, in declaration/execution order."""

    full_reset = EventTerm(
        func=reset_episode_scene,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("target_object")},
    )
    manipulator_reset = EventTerm(
        func=StableOffsetPoseReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "hand_body_name": HAND_BASE_BODY_NAME,
            "c_offset_h": C_OFFSET_H_M,
            "c_quat_h": C_QUAT_H_WXYZ,
            "position_offset_task": _TASK_DEFAULTS.stable_reset_position_offset_task,
            "position_jitter_task": _TASK_DEFAULTS.stable_reset_position_jitter_task,
            "orientation_offset_rpy": _TASK_DEFAULTS.stable_reset_orientation_offset_rpy,
            "orientation_jitter_rpy": _TASK_DEFAULTS.stable_reset_orientation_jitter_rpy,
            "hand_open_range": _TASK_DEFAULTS.stable_reset_hand_open_range,
            "joint_seed_offsets": _TASK_DEFAULTS.stable_reset_joint_seed_offsets,
            "max_pose_attempts": _TASK_DEFAULTS.stable_reset_max_pose_attempts,
            "max_iterations": _TASK_DEFAULTS.stable_reset_max_iterations,
            "damping": _TASK_DEFAULTS.ik_damping,
            "step_size": _TASK_DEFAULTS.ik_step_size,
            "position_error_step": _TASK_DEFAULTS.stable_reset_position_error_step_m,
            "orientation_error_step": _TASK_DEFAULTS.stable_reset_orientation_error_step_rad,
            "position_tolerance": _TASK_DEFAULTS.ik_position_tolerance_m,
            "orientation_tolerance": _TASK_DEFAULTS.ik_orientation_tolerance_rad,
            "singular_value_min": _TASK_DEFAULTS.ik_singular_value_min,
            "joint_limit_margin": _TASK_DEFAULTS.ik_joint_limit_margin_rad,
            "wrist_3_range": _TASK_DEFAULTS.ik_wrist_3_range_rad,
        },
    )


@configclass
class RewardsCfg:
    goal = RewTerm(func=object_goal_reward, weight=4.0)
    actual_normal_alignment = RewTerm(func=actual_contact_normal_alignment, weight=0.75)
    tactile_contact = RewTerm(func=selected_tactile_contact, weight=0.40)
    action_change = RewTerm(func=normalized_action_change, weight=0.06)
    time_cost = RewTerm(func=time_cost_rate, weight=0.20)
    tilt_risk = RewTerm(func=tilt_risk_penalty, weight=1.0)
    board_contact_risk = RewTerm(func=board_contact_risk_penalty, weight=1.0)
    object_height_risk = RewTerm(func=object_height_risk_penalty, weight=1.0)
    success_terminal = RewTerm(func=success_terminal_bonus_rate, weight=1.0)
    failure_terminal = RewTerm(func=failure_terminal_penalty_rate, weight=1.0)


@configclass
class TerminationsCfg:
    # Failure terms are declared before success for readable diagnostics;
    # task_success masks every hard failure on the same step.
    topple = DoneTerm(func=topple_failure)
    robot_board_contact = DoneTerm(func=board_contact_failure)
    height_limit = DoneTerm(func=height_limit_failure)
    out_of_bounds = DoneTerm(func=object_out_of_bounds)
    invalid_reset = DoneTerm(func=invalid_reset_contact)
    success = DoneTerm(func=task_success)
    time_out = DoneTerm(func=episode_timeout, time_out=True)


@configclass
class BlindSweepEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level environment configuration registered by the package."""

    # VisualizationMarkers write CPU-side USD attributes every policy step.
    # Keep them off for large vectorized training; play.py enables them unless
    # explicitly requested otherwise.
    debug_vis: bool = False
    task: BlindSweepTaskCfg = BlindSweepTaskCfg()
    scene: BlindSweepSceneCfg = BlindSweepSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        lazy_sensor_update=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.20,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=5 * 2**17,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 16 * 16,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 16,
            friction_correlation_distance=0.00625,
        ),
    )

    def __post_init__(self):
        self.task.validate()
        self.decimation = 2
        self.episode_length_s = 10.0
        self.is_finite_horizon = False
        self.num_rerenders_on_reset = 0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.50, 2.00, 2.00)
        self.viewer.lookat = (-0.70, 0.0, 1.05)

        # Scene entities are instantiated from class-level defaults, so copy
        # every task-level geometry override into their spawn and initial-state
        # configs before the scene is constructed.  Mutating only these values
        # deliberately preserves all ContactSensor prim/filter paths.
        self.scene.board.spawn.size = self.task.board_size
        self.scene.board.init_state.pos = self.task.board_center
        self.scene.target_object.spawn.size = (self.task.cube_size,) * 3
        board_top = self.task.board_center[2] + 0.5 * self.task.board_size[2]
        cube_initial_center = (
            0.5 * (self.task.object_xy_range_low[0] + self.task.object_xy_range_high[0]),
            0.5 * (self.task.object_xy_range_low[1] + self.task.object_xy_range_high[1]),
            board_top + 0.5 * self.task.cube_size,
        )
        self.scene.target_object.init_state.pos = cube_initial_center
        self.scene.target_object.init_state.rot = self.task.initial_cube_quat_w

        # Synchronize all consumers of T_HC and task-configurable control values
        # when a downstream config subclass overrides BlindSweepTaskCfg.
        self.actions.arm_action.body_offset_pos = self.task.c_offset_h
        self.actions.arm_action.body_offset_quat = self.task.c_quat_h
        self.actions.arm_action.translation_scale = self.task.arm_translation_action_scale_m
        self.actions.arm_action.rotation_scale = self.task.arm_rotation_action_scale_rad
        self.actions.arm_action.motion_stiffness = self.task.osc_motion_stiffness
        self.actions.arm_action.motion_damping_ratio = self.task.osc_motion_damping_ratio
        self.actions.arm_action.effort_limit_scale = self.task.osc_effort_limit_scale
        self.actions.hand_action.max_synergy_rate = self.task.hand_max_synergy_rate_per_s
        self.actions.hand_action.max_joint_target_rate = self.task.hand_max_joint_target_rate_rad_s
        self.events.manipulator_reset.params.update(
            {
                "c_offset_h": self.task.c_offset_h,
                "c_quat_h": self.task.c_quat_h,
                "position_offset_task": self.task.stable_reset_position_offset_task,
                "position_jitter_task": self.task.stable_reset_position_jitter_task,
                "orientation_offset_rpy": self.task.stable_reset_orientation_offset_rpy,
                "orientation_jitter_rpy": self.task.stable_reset_orientation_jitter_rpy,
                "hand_open_range": self.task.stable_reset_hand_open_range,
                "joint_seed_offsets": self.task.stable_reset_joint_seed_offsets,
                "max_pose_attempts": self.task.stable_reset_max_pose_attempts,
                "max_iterations": self.task.stable_reset_max_iterations,
                "damping": self.task.ik_damping,
                "step_size": self.task.ik_step_size,
                "position_error_step": self.task.stable_reset_position_error_step_m,
                "orientation_error_step": self.task.stable_reset_orientation_error_step_rad,
                "position_tolerance": self.task.ik_position_tolerance_m,
                "orientation_tolerance": self.task.ik_orientation_tolerance_rad,
                "singular_value_min": self.task.ik_singular_value_min,
                "joint_limit_margin": self.task.ik_joint_limit_margin_rad,
                "wrist_3_range": self.task.ik_wrist_3_range_rad,
            }
        )


__all__ = ["BlindSweepEnvCfg", "BlindSweepSceneCfg", "BlindSweepTaskCfg"]
