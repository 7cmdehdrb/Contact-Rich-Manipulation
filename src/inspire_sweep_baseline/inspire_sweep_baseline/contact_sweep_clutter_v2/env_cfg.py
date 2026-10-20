"""Clutter v1 with a compensated six-axis observation and one new cost."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.env_cfg import ContactSweepClutterV1EnvCfg
from .observations_cfg import ContactSweepClutterV2ObservationsCfg
from .rewards_cfg import ContactSweepClutterV2RewardsCfg


@configclass
class ContactSweepClutterV2EnvCfg(ContactSweepClutterV1EnvCfg):
    observations: ContactSweepClutterV2ObservationsCfg = ContactSweepClutterV2ObservationsCfg()
    rewards: ContactSweepClutterV2RewardsCfg = ContactSweepClutterV2RewardsCfg()
