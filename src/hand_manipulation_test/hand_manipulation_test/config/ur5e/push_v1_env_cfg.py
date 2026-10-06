"""Inherited Push-v1: realizable palm approach and bounded position-target OSC."""

from copy import deepcopy
import math

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.utils import configclass

from ...assets.push_v1_robot import spawn_push_v1_robot

from ...mdp.push_v1_actions import AccumulatedTranslationOscActionCfg, accumulated_translation_error
from ...mdp.push_v1_commands import CubePushV1CommandCfg
from ...mdp.push_v1_events import PushV1SafePoseReset
from ...mdp.push_v1_rewards import (
    push_v1_action_excess_reward, push_v1_alignment_reward, push_v1_roll_reward,
    push_v1_contact_distance_penalty, push_v1_eef_height_penalty,
)
from .push_env_cfg import (
    PushCommandsCfg, PushEventsCfg, PushObservationsCfg, PushRewardsCfg, PushTaskCfg, UR5eInspirePushEnvCfg,
)


@configclass
class PushV1TaskCfg(PushTaskCfg):
    cube_center_height_m = PushTaskCfg.cube_center_height_m
    object_xy_range_low = PushTaskCfg.object_xy_range_low
    object_xy_range_high = PushTaskCfg.object_xy_range_high
    # Inherit the common tabletop and initial palm clearance. Keep each path
    # centered in the far workspace where both finger axes point outward.
    object_xy_offset_low: tuple[float, float] = (0.02, -0.03)
    object_xy_offset_high: tuple[float, float] = (0.08, 0.03)
    command_centered_path: bool = True
    command_midpoint_x_offset_m: float = 0.05
    command_initial_x_jitter_m: float = 0.002
    # Clearance at reset and the physical approach target are independent.
    contact_palm_height_m: float = 0.025
    left_contact_reference_body: str = "inspire_thumb_force_sensor_4"
    # Nominal, certified far-workspace branches. The inherited three small
    # seed offsets still retry only unsuccessful rows, with the same budget.
    right_reset_arm_seed: tuple[float, ...] = (.179, .154, -1.560, -4.876, 1.750, 1.571)
    left_reset_arm_seed: tuple[float, ...] = (-.615, .159, -1.625, -4.818, .955, -1.571)
    accumulated_position_error_limit_m: float = 0.06
    eef_soft_height_above_table_m: float = 0.15
    eef_hard_height_above_table_m: float = 0.25
    enforce_outward_fingers: bool = True
    minimum_finger_outward_cos: float = 0.25
    wrist_2_branch_sin_margin: float = 0.15

    def validate(self):
        super().validate()
        soft, hard = self.eef_soft_height_above_table_m, self.eef_hard_height_above_table_m
        if not (math.isfinite(soft) and math.isfinite(hard) and 0 <= soft < hard):
            raise ValueError("EEF height limits must be finite and 0 <= soft < hard")
        for name in ("minimum_finger_outward_cos", "wrist_2_branch_sin_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        for name in ("right_reset_arm_seed", "left_reset_arm_seed"):
            seed = getattr(self, name)
            if len(seed) != 6 or not all(math.isfinite(value) for value in seed):
                raise ValueError(f"{name} must contain six finite joint values")
        if self.left_contact_reference_body not in ("inspire_thumb_force_sensor_3", "inspire_thumb_force_sensor_4"):
            raise ValueError("The left contact reference must be an exposed palmar thumb pad")


@configclass
class PushV1CommandsCfg(PushCommandsCfg):
    target_position = CubePushV1CommandCfg(debug_vis=True)


@configclass
class PushV1ObservationsCfg(PushObservationsCfg):
    @configclass
    class PolicyCfg(PushObservationsCfg.PolicyCfg):
        accumulated_translation_error = ObservationTermCfg(func=accumulated_translation_error)

    policy: PolicyCfg = PolicyCfg()


@configclass
class PushV1EventsCfg(PushEventsCfg):
    safe_hand = EventTermCfg(func=PushV1SafePoseReset, mode="reset")


@configclass
class PushV1RewardsCfg(PushRewardsCfg):
    alignment = RewardTermCfg(func=push_v1_alignment_reward, weight=0.3)
    roll = RewardTermCfg(func=push_v1_roll_reward, weight=0.1)
    action_excess = RewardTermCfg(func=push_v1_action_excess_reward, weight=0.02)
    contact_distance = RewardTermCfg(func=push_v1_contact_distance_penalty, weight=1.0)
    eef_height = RewardTermCfg(func=push_v1_eef_height_penalty, weight=5.0)


@configclass
class UR5eInspirePushV1EnvCfg(UR5eInspirePushEnvCfg):
    task: PushV1TaskCfg = PushV1TaskCfg()
    commands: PushV1CommandsCfg = PushV1CommandsCfg()
    observations: PushV1ObservationsCfg = PushV1ObservationsCfg()
    events: PushV1EventsCfg = PushV1EventsCfg()
    rewards: PushV1RewardsCfg = PushV1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        inherited = self.actions.arm_action
        values = {name: deepcopy(value) for name, value in vars(inherited).items() if name != "class_type"}
        values["position_error_limit_m"] = self.task.accumulated_position_error_limit_m
        values["contact_aware_impedance"] = True
        self.actions.arm_action = AccumulatedTranslationOscActionCfg(**values)
        self.scene.robot.spawn.func = spawn_push_v1_robot


__all__ = ["PushV1TaskCfg", "PushV1CommandsCfg", "PushV1ObservationsCfg", "PushV1EventsCfg",
           "PushV1RewardsCfg", "UR5eInspirePushV1EnvCfg"]
