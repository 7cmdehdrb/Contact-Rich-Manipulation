"""Inherit Fusion v2 and override only its curriculum configuration."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v2.env_cfg import ContactSweepClutterFusionV2EnvCfg
from .curriculum_cfg import ContactSweepClutterFusionV3CurriculumCfg


@configclass
class ContactSweepClutterFusionV3EnvCfg(ContactSweepClutterFusionV2EnvCfg):
    curriculum: ContactSweepClutterFusionV3CurriculumCfg = ContactSweepClutterFusionV3CurriculumCfg()
