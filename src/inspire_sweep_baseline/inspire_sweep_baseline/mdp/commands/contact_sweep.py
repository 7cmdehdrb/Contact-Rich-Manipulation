"""Episode-fixed goal command; the marker follows the sampled target."""

from .sweep_goal import SweepGoalCommandCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepCommandsCfg:
    target_goal_pos = SweepGoalCommandCfg(asset_name="object_collection", debug_vis=True)
