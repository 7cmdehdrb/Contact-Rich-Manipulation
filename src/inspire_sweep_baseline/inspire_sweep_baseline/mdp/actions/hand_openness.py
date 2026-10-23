"""Absolute, two-channel Inspire hand control owned by this package."""

from __future__ import annotations

from ..action_functions.hand_openness import HandOpennessAction
from isaaclab.managers import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

@configclass
class HandOpennessActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HandOpennessAction
    asset_name: str = "robot"
