"""Object/support placement, testable without Isaac Sim."""

import numpy as np

DEFAULT_OBJECT_OFFSET = (0.008, 0.050, 0.165)
DEFAULT_OBJECT_LOWERING = 0.020


def object_and_support_positions(hand_pose, object_offset=DEFAULT_OBJECT_OFFSET,
                                 lowering=DEFAULT_OBJECT_LOWERING,
                                 cylinder_height=0.09, support_height=0.04):
    """Lower BOTH bodies along world Z, preserving support-top/object-bottom alignment."""
    if not np.isfinite(lowering) or lowering < 0:
        raise ValueError("Object lowering must be finite and non-negative [m]")
    position = (np.asarray(hand_pose) @ np.array([*object_offset, 1.0]))[:3]
    position[2] -= lowering
    support = position + np.array([0.0, 0.01, -(cylinder_height + support_height) / 2])
    return position, support
