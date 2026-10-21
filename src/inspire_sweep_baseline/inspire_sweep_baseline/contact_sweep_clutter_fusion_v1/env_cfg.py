"""Inherit Fusion v0 and override only curricula and push-fast speed."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.env_cfg import ContactSweepClutterFusionEnvCfg
from .curriculum_cfg import ContactSweepClutterFusionV1CurriculumCfg


@configclass
class ContactSweepClutterFusionV1EnvCfg(ContactSweepClutterFusionEnvCfg):
    curriculum: ContactSweepClutterFusionV1CurriculumCfg = ContactSweepClutterFusionV1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.push_fast.params["speed_condition"] = 1.5
