"""Parameter variant with explicit reward and curriculum configurations."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.env_cfg import ContactSweepClutterFusionV1_1EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_2RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_2CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_2EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_2RewardsCfg = ContactSweepClutterFusionV1_2RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_2CurriculumCfg = ContactSweepClutterFusionV1_2CurriculumCfg()
