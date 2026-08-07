"""Task-specific MDP terms for the basic sweep environment."""

from .actions import (
    OpenGripperRelativeJointPositionAction,
    OpenGripperRelativeJointPositionActionCfg,
)
from .commands import FixedSweepCommand, FixedSweepCommandCfg
from .observations import (
    end_effector_pose_b,
    goal_position_b,
    target_linear_velocity_b,
    target_position_b,
)
from .rewards import (
    MidpointNoEntryPenalty,
    align_ee_target,
    contact_forward_progress,
    lateral_displacement,
    normalized_endpoint_error,
    object_stall_penalty,
    object_velocity_tracking,
    overshoot_penalty,
    pushing_target,
    remaining_horizon_failure_penalty,
    reward_for_hand_reaching,
    shelf_collision,
    stopped_at_goal,
    sweep_success_bonus,
    target_contact_bonus,
)
from .terminations import (
    PushFastTermination,
    object_drop,
    robot_shelf_collision,
    target_reached_and_stopped,
)

__all__ = [
    "FixedSweepCommand",
    "FixedSweepCommandCfg",
    "MidpointNoEntryPenalty",
    "OpenGripperRelativeJointPositionAction",
    "OpenGripperRelativeJointPositionActionCfg",
    "PushFastTermination",
    "align_ee_target",
    "contact_forward_progress",
    "end_effector_pose_b",
    "goal_position_b",
    "lateral_displacement",
    "normalized_endpoint_error",
    "object_drop",
    "object_stall_penalty",
    "object_velocity_tracking",
    "overshoot_penalty",
    "pushing_target",
    "remaining_horizon_failure_penalty",
    "reward_for_hand_reaching",
    "robot_shelf_collision",
    "shelf_collision",
    "stopped_at_goal",
    "sweep_success_bonus",
    "target_contact_bonus",
    "target_position_b",
    "target_linear_velocity_b",
    "target_reached_and_stopped",
]
