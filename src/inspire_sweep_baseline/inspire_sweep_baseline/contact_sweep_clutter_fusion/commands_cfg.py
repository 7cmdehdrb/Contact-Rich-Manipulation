"""The physical goal remains an episode-fixed Cartesian world command."""

from isaaclab.utils import configclass

from ..mdp.commands import SweepGoalCommandCfg


@configclass
class ContactSweepClutterFusionCommandsCfg:
    target_goal_pos = SweepGoalCommandCfg(asset_name="object_collection", debug_vis=True)
