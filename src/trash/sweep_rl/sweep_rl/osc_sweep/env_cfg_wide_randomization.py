"""Wide-domain-randomization copy of the UR5e OSC sweep environment.

This variant intentionally lives in a separate file so the baseline
``env_cfg.py`` remains unchanged. It expands the requested contact force and
randomizes the target-object mass independently for every episode.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import CommandsCfg, EventCfg, UR5eOscSweepEnvCfg


@configclass
class WideRandomizationCommandsCfg(CommandsCfg):
    """Sweep commands covering every planar direction and 8--50 N."""

    desired_motion = mdp.SweepMotionCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        # Full planar circle: every heading in [-pi, pi] is equally likely.
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.10, 0.22),
        force_range=(8.0, 50.0),
        force_tolerance_range=(3.0, 6.0),
        debug_vis=False,
    )


@configclass
class WideRandomizationEventCfg(EventCfg):
    """Baseline resets plus per-episode target mass randomization."""

    randomize_target_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target_object"),
            # ``abs`` samples the final mass directly instead of scaling the
            # baseline cube mass (0.35 kg).
            "mass_distribution_params": (0.3, 3.0),
            "operation": "abs",
            "distribution": "uniform",
            # The range is wide, so inertia must follow the sampled mass.
            "recompute_inertia": True,
        },
    )


@configclass
class UR5eOscSweepWideRandomizationEnvCfg(UR5eOscSweepEnvCfg):
    """Training-only sweep environment with wide mass and force domains."""

    commands: WideRandomizationCommandsCfg = WideRandomizationCommandsCfg()
    events: WideRandomizationEventCfg = WideRandomizationEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Make every arm and gripper joint passive at the actuator PD level.
        # The OSC action continues to command arm torque explicitly.
        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.stiffness = 0.0
            actuator_cfg.damping = 0.0
