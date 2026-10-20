"""Clutter v0 inheritance with asymmetric observations and four added rewards."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter.env_cfg import ContactSweepClutterEnvCfg
from .observations_cfg import ContactSweepClutterV1ObservationsCfg
from .rewards_cfg import ContactSweepClutterV1RewardsCfg


@configclass
class ContactSweepClutterV1EnvCfg(ContactSweepClutterEnvCfg):
    observations: ContactSweepClutterV1ObservationsCfg = ContactSweepClutterV1ObservationsCfg()
    rewards: ContactSweepClutterV1RewardsCfg = ContactSweepClutterV1RewardsCfg()
