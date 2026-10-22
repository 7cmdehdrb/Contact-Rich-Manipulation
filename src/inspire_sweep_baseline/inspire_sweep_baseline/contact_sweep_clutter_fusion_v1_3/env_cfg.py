"""Force curriculum experiment derived from v1_1, with independent managers."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.env_cfg import ContactSweepClutterFusionV1_1EnvCfg
from .rewards_cfg import ContactSweepClutterFusionV1_3RewardsCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_3CurriculumCfg


@configclass
class ContactSweepClutterFusionV1_3EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_3RewardsCfg = ContactSweepClutterFusionV1_3RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_3CurriculumCfg = ContactSweepClutterFusionV1_3CurriculumCfg()
