"""The physical goal remains an episode-fixed Cartesian world command."""

from .sweep_goal import SweepGoalCommandCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterFusionCommandsCfg:
    target_goal_pos = SweepGoalCommandCfg(asset_name="object_collection", debug_vis=True)
