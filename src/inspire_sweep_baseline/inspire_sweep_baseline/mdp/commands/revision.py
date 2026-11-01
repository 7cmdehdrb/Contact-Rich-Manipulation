"""Independent Revision episode goal configuration."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from ..command_functions.revision import RevisionGoalCommand
from ..common.revision_goal import GOAL_SUCCESS_RADIUS


@configclass
class RevisionGoalCommandCfg(CommandTermCfg):
    class_type: type = RevisionGoalCommand
    asset_name: str = "object_collection"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    goal_radius: float = GOAL_SUCCESS_RADIUS


@configclass
class RevisionCommandsCfg:
    target_goal_pos = RevisionGoalCommandCfg(asset_name="object_collection", debug_vis=True)
