"""Episode-fixed object goal command, independent of other project packages."""

from __future__ import annotations

from ..command_functions.sweep_goal import SweepGoalCommand
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

@configclass
class SweepGoalCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = SweepGoalCommand
    asset_name: str = "object_collection"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
