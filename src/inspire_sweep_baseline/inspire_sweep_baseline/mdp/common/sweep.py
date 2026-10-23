"""Self-contained baseline configuration, derived from the v2-mass behavior."""

from ...assets.robot import ARM_JOINT_NAMES
from isaaclab.managers import SceneEntityCfg

TARGET_Z_OFFSET = 0.09  # bottom-origin link; cylinder center is at +0.05 m.


TOOL_ROTATION_H = (0.5, -0.5, -0.5, -0.5)


def _arm():
    return SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)
