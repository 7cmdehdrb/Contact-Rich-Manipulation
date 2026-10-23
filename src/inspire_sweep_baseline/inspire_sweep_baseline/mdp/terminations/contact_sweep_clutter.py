"""Any real object falling or tipping triggers the existing object_drop term."""

from ..termination_functions.contact_sweep_clutter import any_object_fall_or_tilt
from .contact_sweep import ContactSweepTerminationsCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterTerminationsCfg(ContactSweepTerminationsCfg):
    def __post_init__(self):
        self.object_drop.func = any_object_fall_or_tilt
