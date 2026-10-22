"""Parameter variant with explicit reward and curriculum configurations."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.env_cfg import ContactSweepClutterFusionV1_0EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_1RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_1CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_1EnvCfg(ContactSweepClutterFusionV1_0EnvCfg):
    rewards: ContactSweepClutterFusionV1_1RewardsCfg = ContactSweepClutterFusionV1_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1CurriculumCfg = ContactSweepClutterFusionV1_1CurriculumCfg()
