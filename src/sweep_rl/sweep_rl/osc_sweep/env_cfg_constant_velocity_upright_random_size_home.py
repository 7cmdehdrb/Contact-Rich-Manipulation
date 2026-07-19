"""Sweep, park the object, then return the UR5e to Home without contact."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import ARM_ENTITY_CFG, EEF_ENTITY_CFG, OscSweepSceneCfg
from .env_cfg_constant_velocity import ConstantVelocityObservationsCfg
from .env_cfg_constant_velocity_upright_random_size import (
    GripperExclusionRewardsCfg,
    GripperExclusionTerminationsCfg,
    UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg,
)


@configclass
class SweepHomeSceneCfg(OscSweepSceneCfg):
    """Base scene plus target-filtered contact against the entire robot."""

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


@configclass
class SweepHomeCommandsCfg:
    """Constant-speed sweep command with a latched Home-return phase."""

    desired_motion = mdp.SweepHomeConstantVelocityCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.10, 0.22),
        target_speed_range=(0.08, 0.08),
        endpoint_threshold=0.020,
        speed_threshold=0.020,
        goal_dwell_time=0.30,
        debug_vis=False,
    )


@configclass
class SweepHomeObservationsCfg(ConstantVelocityObservationsCfg):
    """ConstantVelocity observations plus the explicit task phase."""

    @configclass
    class PolicyCfg(ConstantVelocityObservationsCfg.PolicyCfg):
        task_phase = ObsTerm(
            func=mdp.task_phase,
            params={"command_name": "desired_motion"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class SweepHomeRewardsCfg(GripperExclusionRewardsCfg):
    """Home-return shaping added to the constant-velocity sweep rewards."""

    home_joint_pose = RewTerm(
        func=mdp.home_joint_pose_reward,
        weight=15.0,
        params={
            "command_name": "desired_motion",
            "joint_std": 0.35,
            "asset_cfg": ARM_ENTITY_CFG,
        },
    )
    home_joint_error = RewTerm(
        func=mdp.home_joint_error,
        weight=-3.0,
        params={
            "command_name": "desired_motion",
            "error_scale": 0.75,
            "maximum_normalized_error": 3.0,
            "asset_cfg": ARM_ENTITY_CFG,
        },
    )
    home_clearance = RewTerm(
        func=mdp.home_eef_object_clearance,
        weight=3.0,
        params={
            "command_name": "desired_motion",
            "safe_distance": 0.22,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    post_goal_contact = RewTerm(
        func=mdp.PhaseGatedReward,
        weight=-12.0,
        params={
            "phase_command_name": "desired_motion",
            "active_phase": 1,
            "reward_func": mdp.filtered_contact_indicator,
            "reward_params": {
                "sensor_name": "target_robot_contact",
                "force_threshold": 0.25,
            },
        },
    )
    goal_hold_error = RewTerm(
        func=mdp.PhaseGatedReward,
        weight=-10.0,
        params={
            "phase_command_name": "desired_motion",
            "active_phase": 1,
            "reward_func": mdp.normalized_endpoint_error,
            "reward_params": {
                "command_name": "desired_motion",
                "maximum_error": 2.0,
                "object_cfg": SceneEntityCfg("target_object"),
            },
        },
    )
    post_goal_object_speed = RewTerm(
        func=mdp.home_object_speed_penalty,
        weight=-3.0,
        params={
            "command_name": "desired_motion",
            "speed_scale": 0.05,
            "maximum_normalized_speed": 4.0,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    post_goal_object_displacement = RewTerm(
        func=mdp.home_object_displacement_penalty,
        weight=-8.0,
        params={
            "command_name": "desired_motion",
            "displacement_scale": 0.010,
            "maximum_normalized_displacement": 4.0,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    home_time = RewTerm(
        func=mdp.home_phase_time,
        weight=-0.5,
        params={"command_name": "desired_motion"},
    )
    home_success = RewTerm(
        func=mdp.home_success_bonus,
        weight=50.0,
        params={
            "command_name": "desired_motion",
            "joint_position_threshold": 0.12,
            "joint_speed_threshold": 0.15,
            "endpoint_threshold": 0.025,
            "object_speed_threshold": 0.025,
            "object_displacement_threshold": 0.010,
            "contact_sensor_name": "target_robot_contact",
            "contact_force_threshold": 0.25,
            "asset_cfg": ARM_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class SweepHomeTerminationsCfg(GripperExclusionTerminationsCfg):
    """End successfully only after a stable, collision-free Home return."""

    success = DoneTerm(
        func=mdp.HomeAfterSweepSuccess,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "joint_position_threshold": 0.12,
            "joint_speed_threshold": 0.15,
            "endpoint_threshold": 0.025,
            "object_speed_threshold": 0.025,
            "object_displacement_threshold": 0.010,
            "dwell_time": 0.25,
            "contact_sensor_name": "target_robot_contact",
            "contact_force_threshold": 0.25,
            "asset_cfg": ARM_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    post_goal_object_moved = DoneTerm(
        func=mdp.object_disturbed_after_sweep,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "displacement_threshold": 0.015,
            "speed_threshold": 0.10,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg(
    UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg
):
    """Two-stage sweep and collision-free Home-return environment."""

    scene: SweepHomeSceneCfg = SweepHomeSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=False,
    )
    commands: SweepHomeCommandsCfg = SweepHomeCommandsCfg()
    observations: SweepHomeObservationsCfg = SweepHomeObservationsCfg()
    rewards: SweepHomeRewardsCfg = SweepHomeRewardsCfg()
    terminations: SweepHomeTerminationsCfg = SweepHomeTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        # The target is the sensor body for whole-robot filtered contact.
        # ContactReporter must therefore be authored on the cube itself.
        self.scene.target_object.spawn.activate_contact_sensors = True

        # These terms define the sweep objective and must stop contributing
        # once the object is parked.  Otherwise the agent can remain in contact
        # at the goal and accumulate the old stopped/success rewards instead of
        # returning Home.
        sweep_only_terms = (
            "push_pose_error",
            "side_direction_error",
            "target_contact",
            "side_center_contact",
            "contact_forward_progress",
            "velocity_tracking",
            "endpoint_error",
            "stopped_at_goal",
            "success",
            "lateral_error",
            "overshoot",
            "stall",
        )
        for term_name in sweep_only_terms:
            term_cfg = getattr(self.rewards, term_name)
            original_func = term_cfg.func
            original_params = dict(term_cfg.params)
            term_cfg.func = mdp.PhaseGatedReward
            term_cfg.params = {
                "phase_command_name": "desired_motion",
                "active_phase": 0,
                "reward_func": original_func,
                "reward_params": original_params,
            }
        failure_names = self.rewards.failure_termination.params["term_names"]
        if "post_goal_object_moved" not in failure_names:
            self.rewards.failure_termination.params["term_names"] = (
                *failure_names,
                "post_goal_object_moved",
            )
