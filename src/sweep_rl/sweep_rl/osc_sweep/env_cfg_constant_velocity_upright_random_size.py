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
from .assets import CONTACT_PAD_SIZE, GRIPPER_SIDE_FACE_NORMAL_AXIS
from .env_cfg import EEF_ENTITY_CFG, EventCfg
from .env_cfg_constant_velocity import (
    ConstantVelocityRewardsCfg,
    UR5eOscSweepConstantVelocityEnvCfg,
)
from .mdp.events import VARIABLE_CUBE_SIZE_BUFFER


@configclass
class UprightRandomSizeRewardsCfg(ConstantVelocityRewardsCfg):
    """Push with the exterior table-side pad while keeping a loose upright pose."""

    push_pose_error = RewTerm(
        func=mdp.variable_size_precontact_pose_error,
        weight=-0.35,
        params={
            "command_name": "desired_motion",
            "distance_scale": 0.10,
            # Preserves the original 0.065 m stand-off for a 0.06 m cube:
            # half-size 0.03 m + free-space clearance 0.035 m.
            "surface_clearance": 0.035,
            # Pads are mounted at local +/-Y = 0.055 m. Raising the EEF by
            # this amount aligns the table-side (-Y) pad with the cube center
            # instead of placing the cube in the open gripper gap.
            "table_side_pad_offset": 0.055,
            "size_buffer_name": VARIABLE_CUBE_SIZE_BUFFER,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    gripper_upright = RewTerm(
        func=mdp.eef_axis_upright_alignment,
        weight=0.35,
        params={
            # Contact pads are mounted at local +/-Y.  Aligning +Y with world
            # +Z puts one side of the U toward the table and the other upward.
            "local_up_axis": (0.0, 1.0, 0.0),
            # Do not distinguish exact vertical from a small natural tilt.
            "allowed_deviation_deg": 15.0,
            "zero_reward_deviation_deg": 60.0,
            "eef_cfg": EEF_ENTITY_CFG,
        },
    )
    target_contact = RewTerm(
        func=mdp.target_contact_bonus,
        weight=0.35,
        # -Y is the table-side pad when the local +Y axis points upward.
        params={"sensor_names": ("right_contact",)},
    )
    side_center_contact = RewTerm(
        func=mdp.side_pad_center_contact,
        weight=1.0,
        params={
            "sensor_names": ("right_contact",),
            "pad_size": CONTACT_PAD_SIZE,
            "face_normal_axis": GRIPPER_SIDE_FACE_NORMAL_AXIS,
            "center_sigma": 0.45,
            "face_sigma": 0.25,
        },
    )
    off_center_contact = RewTerm(
        func=mdp.off_center_target_contact,
        weight=-0.75,
        params={
            "sensor_names": ("right_contact",),
            "pad_size": CONTACT_PAD_SIZE,
            "face_normal_axis": GRIPPER_SIDE_FACE_NORMAL_AXIS,
            "center_sigma": 0.45,
            "face_sigma": 0.25,
        },
    )
    dual_pad_contact = RewTerm(
        func=mdp.dual_pad_target_contact,
        weight=-2.0,
        params={"sensor_names": ("left_contact", "right_contact")},
    )
    object_in_gap = RewTerm(
        func=mdp.object_in_gripper_gap,
        weight=-2.0,
        params={
            "gap_axis": 1,
            # Pad centers are +/-0.055 m and their Y thickness is 0.030 m,
            # leaving an approximately +/-0.040 m center gap.
            "gap_half_width": 0.040,
            "transition_width": 0.015,
            "proximity_scale": 0.12,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    contact_forward_progress = RewTerm(
        func=mdp.contact_forward_progress,
        weight=3.0,
        params={
            "command_name": "desired_motion",
            "acceleration_distance": 0.025,
            "stopping_distance": 0.04,
            "initial_speed_fraction": 0.25,
            "endpoint_threshold": 0.020,
            "maximum_normalized_speed": 1.25,
            "sensor_names": ("right_contact",),
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    velocity_tracking = RewTerm(
        func=mdp.side_contact_velocity_profile_tracking,
        weight=10.0,
        params={
            "command_name": "desired_motion",
            "std": 0.025,
            "acceleration_distance": 0.025,
            "stopping_distance": 0.04,
            "initial_speed_fraction": 0.25,
            "endpoint_threshold": 0.020,
            "sensor_names": ("right_contact",),
            "pad_size": CONTACT_PAD_SIZE,
            "face_normal_axis": GRIPPER_SIDE_FACE_NORMAL_AXIS,
            "center_sigma": 0.45,
            "face_sigma": 0.25,
            "object_cfg": SceneEntityCfg("target_object"),
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
