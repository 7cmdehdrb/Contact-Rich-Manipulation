"""Any real object falling or tipping triggers the existing object_drop term."""

from isaaclab.utils import configclass

from ..contact_sweep.terminations_cfg import ContactSweepTerminationsCfg
from .mdp import any_object_fall_or_tilt


@configclass
class ContactSweepClutterTerminationsCfg(ContactSweepTerminationsCfg):
    def __post_init__(self):
        self.object_drop.func = any_object_fall_or_tilt
