"""Preserve the checkpoint's policy, rewards, curricula and physics contracts."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1_curriculum.env_cfg import ContactSweepClutterV1CurriculumEnvCfg


@configclass
class ContactSweepClutterFTTestEnvCfg(ContactSweepClutterV1CurriculumEnvCfg):
    """The v2 compensation runs outside policy observations and rewards."""
