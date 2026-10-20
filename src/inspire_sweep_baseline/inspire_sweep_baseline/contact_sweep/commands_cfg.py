"""Episode-fixed goal command; the marker follows the sampled target."""

from isaaclab.utils import configclass
from ..mdp.commands import SweepGoalCommandCfg


@configclass
class ContactSweepCommandsCfg:
    target_goal_pos = SweepGoalCommandCfg(asset_name="object_collection", debug_vis=True)
