"""Horizontal Cube pushing on the inherited physical table and palmar pads."""

import math

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import mdp
from ...env_cfg import ObservationsCfg
from ...mdp.push_commands import CubePushCommandCfg
from ...mdp.push_events import PushSafePoseReset, reset_cube_for_push
from ...mdp.rewards import ExecutedActionRate
from ...push_math import cube_circumsphere_radius
from .contact_env_cfg import (
    ContactCommandsCfg,
    ContactEventsCfg,
    ContactRewardsCfg,
    ContactTaskCfg,
    ContactTerminationsCfg,
    UR5eInspireContactEnvCfg,
    UR5eInspireContactSceneCfg,
)


@configclass
class PushTaskCfg(ContactTaskCfg):
    # Isaac Lab's configclass post-init detects properties on the immediate
    # class; redeclare inherited read-only properties for that check.
    cube_center_height_m = ContactTaskCfg.cube_center_height_m
    object_xy_range_low = ContactTaskCfg.object_xy_range_low
    object_xy_range_high = ContactTaskCfg.object_xy_range_high
    object_xy_offset_low: tuple[float, float] = (0.04, -0.06)
    object_xy_offset_high: tuple[float, float] = (0.06, 0.06)
    command_distance_range: tuple[float, float] = (0.20, 0.30)
    command_angle_jitter_rad: float = math.pi / 18.0
    command_sample_attempts: int = 128
    table_path_margin_m: float = 0.01
    table_failure_margin_m: float = 0.005
    # Inherit the common 10cm start height so the outward-facing thumb clears
    # the table on both sides. Push applies this offset to the physical pad.
    reset_position_jitter_task: tuple[float, float, float] = (0.004, 0.004, 0.003)
    # Keep a bounded three-seed search with both wrist branches.
    reset_joint_seed_offsets: tuple[tuple[float, ...], ...] = (
        (0.0, 1.2, -4.4, 0.0, 0.0, 0.0),
        (0.0, 1.2, -4.4, math.pi, 0.0, 0.0),
        (-2.0, 0.85, -4.4, 3.55, -1.87, 0.785),
    )
    side_contact_angle_rad: float = math.pi / 6.0
    success_distance_m: float = 0.01
    success_speed_m_s: float = 0.02
    success_hold_time_s: float = 0.20
    # Shape normalized forward progress: smaller values emphasize early motion.
    progress_sigma_fraction: float = 0.5

    def validate(self):
        super().validate()
        for name in ("success_speed_m_s", "success_hold_time_s", "progress_sigma_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            len(self.command_distance_range) != 2
            or not all(math.isfinite(value) for value in self.command_distance_range)
            or not 0.0 < self.command_distance_range[0] <= self.command_distance_range[1]
        ):
            raise ValueError("command_distance_range must be positive and increasing")
        if not math.isfinite(self.command_angle_jitter_rad) or not 0.0 <= self.command_angle_jitter_rad < math.pi / 2:
            raise ValueError("command_angle_jitter_rad must be within [0, pi/2)")
        if not isinstance(self.command_sample_attempts, int) or self.command_sample_attempts <= 0:
            raise ValueError("command_sample_attempts must be a positive integer")
        if not math.isfinite(self.side_contact_angle_rad) or not 0.0 < self.side_contact_angle_rad < math.pi / 2:
            raise ValueError("side_contact_angle_rad must be within (0, pi/2)")
        if (
            not math.isfinite(self.table_path_margin_m)
            or not math.isfinite(self.table_failure_margin_m)
            or not 0.0 <= self.table_failure_margin_m <= self.table_path_margin_m
        ):
            raise ValueError("Table margins must be finite, nonnegative, and path margin must cover failure margin")
        radius = cube_circumsphere_radius(self.cube_size) + self.table_path_margin_m
        for axis in range(2):
            surface_low = self.table_center[axis] - self.table_size[axis] / 2 + radius
            surface_high = self.table_center[axis] + self.table_size[axis] / 2 - radius
            if surface_low > surface_high:
                raise ValueError("The table cannot contain an orientation-safe Cube footprint")
            if self.object_xy_range_low[axis] < surface_low - 1e-8 or self.object_xy_range_high[axis] > surface_high + 1e-8:
                raise ValueError("Push Cube sampling range must fit the orientation-safe table rectangle")


@configclass
class UR5eInspirePushSceneCfg(UR5eInspireContactSceneCfg):
    cube_table_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        update_period=0.0,
        history_length=2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        max_contact_data_count_per_prim=128,
        debug_vis=False,
    )


@configclass
class PushCommandsCfg(ContactCommandsCfg):
    target_position = CubePushCommandCfg(debug_vis=True)


@configclass
class PushObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        push_command = ObservationTermCfg(func=mdp.push_command_observation)
        current_cube_base_position = ObservationTermCfg(func=mdp.current_cube_base_position)

    policy: PolicyCfg = PolicyCfg()


@configclass
class PushEventsCfg(ContactEventsCfg):
    reset_cube = EventTermCfg(func=reset_cube_for_push, mode="reset")
    safe_hand = EventTermCfg(func=PushSafePoseReset, mode="reset")


@configclass
class PushRewardsCfg(ContactRewardsCfg):
    eef_distance = None
    palm_cube_contact = None
    approach = RewardTermCfg(func=mdp.push_approach_reward, weight=0.05)
    progress = RewardTermCfg(func=mdp.push_progress_reward, weight=12.0)
    backslide = RewardTermCfg(func=mdp.push_backslide_reward, weight=6.0)
    first_contact = RewardTermCfg(func=mdp.push_first_contact_reward, weight=0.02)
    contact = RewardTermCfg(func=mdp.push_contact_reward, weight=0.005)
    alignment = RewardTermCfg(func=mdp.push_alignment_reward, weight=0.3)
    action_rate = RewardTermCfg(func=ExecutedActionRate, weight=0.001)
    success = RewardTermCfg(func=mdp.push_success_reward, weight=2.0)
    failure = RewardTermCfg(func=mdp.push_failure_reward, weight=2.0)


@configclass
class PushTerminationsCfg(ContactTerminationsCfg):
    table_contact = None
    time_out = TerminationTermCfg(func=mdp.push_time_out, time_out=True)
    success = TerminationTermCfg(func=mdp.push_success)
    failure = TerminationTermCfg(func=mdp.push_failure)


@configclass
class UR5eInspirePushEnvCfg(UR5eInspireContactEnvCfg):
    task: PushTaskCfg = PushTaskCfg()
    scene: UR5eInspirePushSceneCfg = UR5eInspirePushSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, lazy_sensor_update=False,
    )
    commands: PushCommandsCfg = PushCommandsCfg()
    observations: PushObservationsCfg = PushObservationsCfg()
    events: PushEventsCfg = PushEventsCfg()
    rewards: PushRewardsCfg = PushRewardsCfg()
    terminations: PushTerminationsCfg = PushTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.motion_stiffness = (200.0,) * 6
        self.actions.hand_action.synergy_range = (0.8, 1.0)
        self.actions.hand_action.enforce_synergy_joint_limits = True
        self.scene.cube_palm_contacts.history_length = self.decimation
        self.scene.cube_table_contacts.history_length = self.decimation


__all__ = [
    "PushTaskCfg", "UR5eInspirePushSceneCfg", "PushCommandsCfg", "PushObservationsCfg", "PushEventsCfg",
    "PushRewardsCfg", "PushTerminationsCfg", "UR5eInspirePushEnvCfg",
]
