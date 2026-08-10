"""Local MDP API for the standalone shelf Cube Sweep v5 task."""

from .actions import (
    ArmJointPositionWithOpenGripperAction,
    ArmJointPositionWithOpenGripperActionCfg,
)
from .assets import BottomHeavyCuboidCfg, spawn_bottom_heavy_cuboid
from .commands import CubeSweepPoseCommand, CubeSweepPoseCommandCfg
from .observations import (
    cube_width,
    current_ee_pose_b,
    sweep_goal_position_b,
    target_object_position_b,
)
from .rewards import (
    PushingTargetDeltaReward,
    align_ee_to_shelf,
    cube_push_offset_position,
    cube_upright_quality,
    homing_reward,
    pushing_target_raw_reward,
    shelf_collision,
    shelf_floor_contact_mask,
    signed_axis_alignment,
    tcp_position_command_reward_exp,
)
from .terminations import (
    arm_joint_too_fast,
    cube_below_minimum_height,
    cube_excessive_tilt,
    cube_tilt_angle,
    cube_too_fast,
)

__all__ = [
    "BottomHeavyCuboidCfg",
    "CubeSweepPoseCommand",
    "CubeSweepPoseCommandCfg",
    "ArmJointPositionWithOpenGripperAction",
    "ArmJointPositionWithOpenGripperActionCfg",
    "PushingTargetDeltaReward",
    "align_ee_to_shelf",
    "arm_joint_too_fast",
    "cube_below_minimum_height",
    "cube_excessive_tilt",
    "cube_push_offset_position",
    "cube_tilt_angle",
    "cube_too_fast",
    "cube_upright_quality",
    "cube_width",
    "current_ee_pose_b",
    "homing_reward",
    "pushing_target_raw_reward",
    "shelf_collision",
    "shelf_floor_contact_mask",
    "signed_axis_alignment",
    "spawn_bottom_heavy_cuboid",
    "sweep_goal_position_b",
    "target_object_position_b",
    "tcp_position_command_reward_exp",
]
