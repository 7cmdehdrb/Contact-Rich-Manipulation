"""Goal and contact-width experiment derived from v1_3, with independent managers."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_3.env_cfg import ContactSweepClutterFusionV1_3EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_3_1RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_3_1CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_3_1EnvCfg(ContactSweepClutterFusionV1_3EnvCfg):
    rewards: ContactSweepClutterFusionV1_3_1RewardsCfg = ContactSweepClutterFusionV1_3_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_3_1CurriculumCfg = ContactSweepClutterFusionV1_3_1CurriculumCfg()
