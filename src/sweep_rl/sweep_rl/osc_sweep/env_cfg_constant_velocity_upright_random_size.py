"""Upright-gripper, random-cube-size variant of ConstantVelocity-v0.

The original constant-velocity environment is intentionally left unchanged.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import EEF_ENTITY_CFG, EventCfg
from .env_cfg_constant_velocity import (
    ConstantVelocityRewardsCfg,
    UR5eOscSweepConstantVelocityEnvCfg,
)
from .mdp.events import VARIABLE_CUBE_SIZE_BUFFER


@configclass
class UprightRandomSizeRewardsCfg(ConstantVelocityRewardsCfg):
    """Constant-velocity rewards plus a mild vertical U-shape preference."""

    push_pose_error = RewTerm(
        func=mdp.variable_size_precontact_pose_error,
        weight=-0.35,
        params={
            "command_name": "desired_motion",
            "distance_scale": 0.10,
            # Preserves the original 0.065 m stand-off for a 0.06 m cube:
            # half-size 0.03 m + free-space clearance 0.035 m.
            "surface_clearance": 0.035,
            "size_buffer_name": VARIABLE_CUBE_SIZE_BUFFER,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    gripper_upright = RewTerm(
        func=mdp.eef_axis_upright_alignment,
        weight=0.75,
        params={
            # Contact pads are mounted at local +/-Y.  Aligning +Y with world
            # +Z puts one side of the U toward the table and the other upward.
            "local_up_axis": (0.0, 1.0, 0.0),
            "eef_cfg": EEF_ENTITY_CFG,
        },
    )


@configclass
class UprightRandomSizeEventCfg(EventCfg):
    """Baseline resets with per-environment 0.06--0.12 m cube sizes."""

    randomize_target_size = EventTerm(
        func=mdp.randomize_target_cube_size,
        mode="prestartup",
        params={
            "size_range": (0.06, 0.12),
            "base_size": 0.06,
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )
    reset_target = EventTerm(
        func=mdp.reset_variable_size_target_object,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            # Table center 0.75 m + half-height 0.025 m.
            "table_top_height": 0.775,
            "clearance": 0.0,
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg(
    UR5eOscSweepConstantVelocityEnvCfg
):
    """Constant velocity task with upright tool and randomized cube size."""

    rewards: UprightRandomSizeRewardsCfg = UprightRandomSizeRewardsCfg()
    events: UprightRandomSizeEventCfg = UprightRandomSizeEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Required for independent USD/physics scale per parallel environment.
        self.scene.replicate_physics = False
