"""MDP additions for the inherited shelf Cube sweep task."""

from .actions import (
    RateLimitedJointPositionToLimitsAction,
    RateLimitedJointPositionToLimitsActionCfg,
)
from .assets import BottomHeavyCuboidCfg, spawn_bottom_heavy_cuboid
from .commands import CubeSweepPoseCommand, CubeSweepPoseCommandCfg
from .observations import end_effector_pose_b
from .rewards import (
    cube_upright_quality,
    cube_push_offset_position,
    pushing_target,
    pushing_target_raw_reward,
)
from .self_collision import (
    SelfCollisionTermination,
    self_collision_force_penalty,
    self_collision_terminal_penalty,
)

__all__ = [
    "BottomHeavyCuboidCfg",
    "CubeSweepPoseCommand",
    "CubeSweepPoseCommandCfg",
    "RateLimitedJointPositionToLimitsAction",
    "RateLimitedJointPositionToLimitsActionCfg",
    "SelfCollisionTermination",
    "cube_upright_quality",
    "cube_push_offset_position",
    "end_effector_pose_b",
    "pushing_target",
    "pushing_target_raw_reward",
    "self_collision_force_penalty",
    "self_collision_terminal_penalty",
    "spawn_bottom_heavy_cuboid",
]
