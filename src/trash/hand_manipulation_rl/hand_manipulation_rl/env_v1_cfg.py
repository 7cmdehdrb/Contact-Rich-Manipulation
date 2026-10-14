"""Configuration for the corrected inherited Approach-v1 environment."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .env_approach_cfg import (
    BlindSweepApproachEnvCfg,
    BlindSweepApproachRewardsCfg,
    BlindSweepApproachTaskCfg,
)
from .env_cfg import ObservationsCfg
from .mdp.actions import TaskFrameOscAction
from .mdp.v1_observations import approach_v1_policy_observation
from .mdp.v1_rewards import (
    SelectedSurfaceApproachProgress,
    TargetHandObjectContactAcquisition,
    selected_surface_height_safety,
)


@configclass
class BlindSweepApproachV1TaskCfg(BlindSweepApproachTaskCfg):
    """Safer exploration scales and a shorter collision-certified approach."""

    stable_reset_position_offset_task: tuple[float, float, float] = (0.100, 0.0, 0.100)

    # TaskFrameOscAction interprets these as
    # (world-up, toward-object, world-X shelf-depth).
    arm_translation_action_scale_m: tuple[float, float, float] = (0.004, 0.012, 0.004)
    arm_rotation_action_scale_rad: tuple[float, float, float] = (0.030, 0.030, 0.030)

    # The forward shaping frontier is beyond the old upstream-face optimum,
    # but still short enough that a hand passing a stationary Cube cannot earn
    # the complete push return. Actual object progress unlocks the remainder.
    approach_free_distance_palm_hand_x_up_m: float = 0.105
    approach_free_distance_palm_hand_x_down_m: float = 0.055
    approach_free_distance_dorsal_hand_x_up_m: float = 0.105
    approach_free_distance_dorsal_hand_x_down_m: float = 0.165
    approach_precontact_potential_fraction: float = 0.35
    approach_path_cross_track_weight: float = 1.0
    approach_path_height_weight: float = 1.5

    # Measured board-safe diagonal paths for palm/dorsal x Hand-X up/down.
    approach_contact_drop_palm_hand_x_up_m: float = 0.014
    approach_contact_drop_palm_hand_x_down_m: float = 0.006
    approach_contact_drop_dorsal_hand_x_up_m: float = 0.020
    approach_contact_drop_dorsal_hand_x_down_m: float = 0.035
    approach_target_contact_threshold_n: float = 0.05
    approach_contact_push_force_n: float = 10.0
    approach_contact_push_force_guard_n: float = 12.0
    approach_contact_push_board_guard_n: float = 2.0

    safe_height_below_contact_path_m: float = 0.006
    safe_height_penalty_band_m: float = 0.015


@configclass
class BlindSweepApproachV1ObservationsCfg(ObservationsCfg):
    """The inherited 57-D vector plus explicit 3-D approach error."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        vector = ObsTerm(func=approach_v1_policy_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BlindSweepApproachV1RewardsCfg(BlindSweepApproachRewardsCfg):
    """Contact-coupled progress, one-shot contact, and height safety."""

    # The asset's palm pads never contact the Cube on the calibrated approach,
    # whereas the dorsal approximation reads carrier links. Keeping this term
    # would therefore create a mode-dependent bonus. Bits remain observable.
    tactile_contact = None
    # Disable v0's upstream-face occupancy reward: calibration showed that it
    # peaks before real contact and then penalizes continuing into the Cube.
    selected_surface_approach = None
    selected_surface_approach_progress = RewTerm(
        func=SelectedSurfaceApproachProgress,
        weight=2.0,
    )
    target_hand_contact_acquisition = RewTerm(
        func=TargetHandObjectContactAcquisition,
        weight=0.50,
    )
    selected_surface_height_safety = RewTerm(
        func=selected_surface_height_safety,
        weight=1.0,
    )


@configclass
class BlindSweepApproachV1EnvCfg(BlindSweepApproachEnvCfg):
    """Approach-v0 inheritance with corrected control and reward semantics."""

    task: BlindSweepApproachV1TaskCfg = BlindSweepApproachV1TaskCfg()
    observations: BlindSweepApproachV1ObservationsCfg = (
        BlindSweepApproachV1ObservationsCfg()
    )
    rewards: BlindSweepApproachV1RewardsCfg = BlindSweepApproachV1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.class_type = TaskFrameOscAction
        self.actions.arm_action.gravity_compensation = True
        self.actions.arm_action.contact_push_force_n = (
            self.task.approach_contact_push_force_n
        )
        self.actions.arm_action.contact_push_force_guard_n = (
            self.task.approach_contact_push_force_guard_n
        )
        self.actions.arm_action.contact_push_board_guard_n = (
            self.task.approach_contact_push_board_guard_n
        )

        positive_values = {
            "approach_free_distance_palm_hand_x_up_m": (
                self.task.approach_free_distance_palm_hand_x_up_m
            ),
            "approach_free_distance_palm_hand_x_down_m": (
                self.task.approach_free_distance_palm_hand_x_down_m
            ),
            "approach_free_distance_dorsal_hand_x_up_m": (
                self.task.approach_free_distance_dorsal_hand_x_up_m
            ),
            "approach_free_distance_dorsal_hand_x_down_m": (
                self.task.approach_free_distance_dorsal_hand_x_down_m
            ),
            "approach_path_cross_track_weight": (
                self.task.approach_path_cross_track_weight
            ),
            "approach_path_height_weight": self.task.approach_path_height_weight,
            "approach_contact_drop_palm_hand_x_up_m": (
                self.task.approach_contact_drop_palm_hand_x_up_m
            ),
            "approach_contact_drop_palm_hand_x_down_m": (
                self.task.approach_contact_drop_palm_hand_x_down_m
            ),
            "approach_contact_drop_dorsal_hand_x_up_m": (
                self.task.approach_contact_drop_dorsal_hand_x_up_m
            ),
            "approach_contact_drop_dorsal_hand_x_down_m": (
                self.task.approach_contact_drop_dorsal_hand_x_down_m
            ),
            "approach_target_contact_threshold_n": (
                self.task.approach_target_contact_threshold_n
            ),
            "approach_contact_push_force_n": (
                self.task.approach_contact_push_force_n
            ),
            "approach_contact_push_force_guard_n": (
                self.task.approach_contact_push_force_guard_n
            ),
            "approach_contact_push_board_guard_n": (
                self.task.approach_contact_push_board_guard_n
            ),
            "safe_height_below_contact_path_m": (
                self.task.safe_height_below_contact_path_m
            ),
            "safe_height_penalty_band_m": self.task.safe_height_penalty_band_m,
            "arm_vertical_action_scale_m": self.task.arm_translation_action_scale_m[0],
            "arm_approach_action_scale_m": self.task.arm_translation_action_scale_m[1],
            "arm_depth_action_scale_m": self.task.arm_translation_action_scale_m[2],
        }
        invalid = [
            name
            for name, value in positive_values.items()
            if value <= 0.0 or not math.isfinite(value)
        ]
        if invalid:
            raise ValueError(f"Approach-v1 values must be finite and positive: {invalid}")
        if not 0.0 < self.task.approach_precontact_potential_fraction < 1.0:
            raise ValueError("approach_precontact_potential_fraction must be in (0, 1)")


__all__ = [
    "BlindSweepApproachV1EnvCfg",
    "BlindSweepApproachV1ObservationsCfg",
    "BlindSweepApproachV1RewardsCfg",
    "BlindSweepApproachV1TaskCfg",
]
