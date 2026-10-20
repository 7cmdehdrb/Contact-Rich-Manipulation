"""Only the existing tilt penalty expands from the target to all objects."""

from isaaclab.utils import configclass

from ..contact_sweep.rewards_cfg import ContactSweepRewardsCfg
from .mdp import all_object_tilt_cost


@configclass
class ContactSweepClutterRewardsCfg(ContactSweepRewardsCfg):
    def __post_init__(self):
        self.object_tilt.func = all_object_tilt_cost
