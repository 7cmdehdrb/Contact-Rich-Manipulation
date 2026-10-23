"""Relative OSC commands with separate physical translation/rotation bounds."""

from __future__ import annotations

from ..action_functions.bounded_osc import BoundedOperationalSpaceControllerAction
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

@configclass
class BoundedOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = BoundedOperationalSpaceControllerAction
    position_command_limit: float = 0.005
    """Maximum translation delta norm per policy command, in meters."""
    orientation_command_limit: float = 0.025
    """Maximum rotation-vector norm per policy command, in radians."""
