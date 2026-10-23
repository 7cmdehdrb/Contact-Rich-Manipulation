"""Geometry and named joint selectors owned by the fusion task."""

from ...assets.robot import ARM_JOINT_NAMES
from isaaclab.managers import SceneEntityCfg

OBJECT_RADIUS = 0.04


SHELF_HALF_EXTENTS = (0.20, 0.50)


SHELF_SURFACE_HEIGHT = 1.05


OBJECT_MASS = 0.5475


OBJECT_MASS_SCALE = 1.5


TOOL_ROTATION_H = (0.5, -0.5, -0.5, -0.5)


def arm_entity():
    return SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)
