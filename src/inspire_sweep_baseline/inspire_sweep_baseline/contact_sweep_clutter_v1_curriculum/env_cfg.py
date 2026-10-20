"""Clutter v1 plus four reward curricula."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.env_cfg import ContactSweepClutterV1EnvCfg
from .curriculum_cfg import ContactSweepClutterV1CurriculumCfg
from .rewards_cfg import ContactSweepClutterV1CurriculumRewardsCfg


@configclass
class ContactSweepClutterV1CurriculumEnvCfg(ContactSweepClutterV1EnvCfg):
    rewards: ContactSweepClutterV1CurriculumRewardsCfg = ContactSweepClutterV1CurriculumRewardsCfg()
    curriculum: ContactSweepClutterV1CurriculumCfg = ContactSweepClutterV1CurriculumCfg()
