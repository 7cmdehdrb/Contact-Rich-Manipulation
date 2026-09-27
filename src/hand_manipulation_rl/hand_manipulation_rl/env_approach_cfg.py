"""Configuration for the inherited, approach-shaped blind-sweep variant."""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .env_cfg import BlindSweepEnvCfg, BlindSweepTaskCfg, RewardsCfg
from .mdp.approach_events import OrientationAwareStableOffsetPoseReset
from .mdp.approach_rewards import selected_surface_object_approach


@configclass
class BlindSweepApproachTaskCfg(BlindSweepTaskCfg):
    """Parent task values plus collision-aware lower reset heights."""

    # This scalar is the collision-safe fallback passed through the parent reset API.
    # OrientationAwareStableOffsetPoseReset replaces it per environment while
    # preserving the parent's +/-3 mm sampling jitter.
    stable_reset_position_offset_task: tuple[float, float, float] = (0.140, 0.0, 0.100)
    stable_reset_hand_x_up_height_m: float = 0.065
    stable_reset_hand_x_down_height_m: float = 0.100

    # Selected-surface proxy to upstream Cube-face shaping.  The vertical
    # scale is intentionally tighter than the planar scale so moving down
    # toward the board immediately loses approach reward.
    approach_planar_capture_radius_m: float = 0.015
    approach_planar_sigma_m: float = 0.080
    approach_height_sigma_m: float = 0.030


@configclass
class BlindSweepApproachRewardsCfg(RewardsCfg):
    """Parent rewards plus a small dense pre-contact/maintenance signal."""

    selected_surface_approach = RewTerm(
        func=selected_surface_object_approach,
        weight=0.10,
    )


@configclass
class BlindSweepApproachEnvCfg(BlindSweepEnvCfg):
    """Inherited environment config; the original v0 config is unchanged."""

    task: BlindSweepApproachTaskCfg = BlindSweepApproachTaskCfg()
    rewards: BlindSweepApproachRewardsCfg = BlindSweepApproachRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.manipulator_reset.func = OrientationAwareStableOffsetPoseReset

        positive_values = {
            "stable_reset_hand_x_up_height_m": self.task.stable_reset_hand_x_up_height_m,
            "stable_reset_hand_x_down_height_m": self.task.stable_reset_hand_x_down_height_m,
            "approach_planar_sigma_m": self.task.approach_planar_sigma_m,
            "approach_height_sigma_m": self.task.approach_height_sigma_m,
        }
        invalid = [
            name
            for name, value in positive_values.items()
            if value <= 0.0 or not math.isfinite(value)
        ]
        if invalid:
            raise ValueError(f"Approach task values must be finite and positive: {invalid}")
        if (
            self.task.approach_planar_capture_radius_m < 0.0
            or not math.isfinite(self.task.approach_planar_capture_radius_m)
        ):
            raise ValueError("approach_planar_capture_radius_m must be finite and non-negative")
        if min(
            self.task.stable_reset_hand_x_up_height_m,
            self.task.stable_reset_hand_x_down_height_m,
        ) <= 0.5 * self.task.cube_size:
            raise ValueError("Approach reset heights must remain above the Cube half extent")


__all__ = [
    "BlindSweepApproachEnvCfg",
    "BlindSweepApproachRewardsCfg",
    "BlindSweepApproachTaskCfg",
]
