"""Self-contained baseline configuration, derived from the v2-mass behavior."""

from .sweep_goal import SweepGoalCommandCfg
from isaaclab.utils import configclass

@configclass
class CommandsCfg:
    target_goal_pos = SweepGoalCommandCfg(asset_name="object_collection")
