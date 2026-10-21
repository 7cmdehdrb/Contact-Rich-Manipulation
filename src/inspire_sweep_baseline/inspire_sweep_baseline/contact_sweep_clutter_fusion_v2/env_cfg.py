"""Preserve Fusion v0 and disable only its EEF orientation reward term."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.env_cfg import ContactSweepClutterFusionEnvCfg


@configclass
class ContactSweepClutterFusionV2EnvCfg(ContactSweepClutterFusionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.eef_orientation = None
