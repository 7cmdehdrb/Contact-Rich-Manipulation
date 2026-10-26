"""Independent Final v0 command Cfgs using the episode-fixed goal implementation."""


from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from ..command_functions.sweep_goal import SweepGoalCommand



@configclass
class FinalGoalCommandCfg(CommandTermCfg):
    class_type: type = SweepGoalCommand
    asset_name: str = "object_collection"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False



@configclass
class FinalCommandsCfg:
    target_goal_pos = FinalGoalCommandCfg(asset_name="object_collection", debug_vis=True)
