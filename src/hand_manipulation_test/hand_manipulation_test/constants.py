"""Immutable action, observation and initial-state contracts."""

from types import MappingProxyType


ACTION_DIM = 8
C_OFFSET_H_M = (0.0, 0.05, 0.10)
C_QUAT_H_WXYZ = (1.0, 0.0, 0.0, 0.0)
INITIAL_ARM_JOINT_POSITIONS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -2.2,
    "elbow_joint": 2.2,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.785,
}
INITIAL_HAND_SYNERGY = (0.5, 0.5)
OBSERVATION_LAYOUT = (
    ("arm_joint_position", 6),
    ("arm_joint_velocity", 6),
    ("eef_relative_position", 3),
    ("eef_relative_orientation", 3),
    ("hand_state", 2),
    ("surface_header_and_tactile", 18),
    ("wrist_wrench_c", 6),
    ("initial_target_relative_position", 3),
    ("last_action", 8),
)


def _observation_slices(layout=OBSERVATION_LAYOUT):
    offset = 0
    result = {}
    for name, width in layout:
        result[name] = slice(offset, offset + width)
        offset += width
    return MappingProxyType(result)


OBSERVATION_SLICES = _observation_slices()
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_LAYOUT)

PUSH_OBSERVATION_LAYOUT = OBSERVATION_LAYOUT + (
    ("push_command", 3),
    ("current_cube_base_position", 3),
)
PUSH_OBSERVATION_SLICES = _observation_slices(PUSH_OBSERVATION_LAYOUT)
PUSH_OBSERVATION_DIM = sum(width for _, width in PUSH_OBSERVATION_LAYOUT)
