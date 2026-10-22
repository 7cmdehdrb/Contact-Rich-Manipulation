"""Force curriculum experiment derived from v1_1, with independent managers."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.env_cfg import ContactSweepClutterFusionV1_1EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_1_2RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_1_2CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_1_2EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_1_2RewardsCfg = ContactSweepClutterFusionV1_1_2RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1_2CurriculumCfg = ContactSweepClutterFusionV1_1_2CurriculumCfg()
