"""Inherit Fusion v0, remove EEF orientation cost, and raise push-fast speed."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.env_cfg import ContactSweepClutterFusionEnvCfg


@configclass
class ContactSweepClutterFusionV2EnvCfg(ContactSweepClutterFusionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.eef_orientation = None
        self.terminations.push_fast.params["speed_condition"] = 1.5
