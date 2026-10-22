"""Acceleration experiment derived from v1_1, with independent managers."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.env_cfg import ContactSweepClutterFusionV1_1EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_1_1RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_1_1CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_1_1EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_1_1RewardsCfg = ContactSweepClutterFusionV1_1_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1_1CurriculumCfg = ContactSweepClutterFusionV1_1_1CurriculumCfg()
