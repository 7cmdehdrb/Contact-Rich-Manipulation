"""Clutter v1 rewards plus a small, strictly budgeted wrench-change penalty."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.rewards_cfg import ContactSweepClutterV1RewardsCfg
from .mdp import WrenchChangePenalty


@configclass
class ContactSweepClutterV2RewardsCfg(ContactSweepClutterV1RewardsCfg):
    # Must remain after the inherited positive terms: their current-step sums
    # supply the budget. At 50 Hz, ignore changes <=0.5 N and <=0.01 N m.
    # Max -0.1/s, hence -1 over 10 s, AND <=5% of actual gross positive return.
    wrench_change = RewardTermCfg(
        func=WrenchChangePenalty,
        weight=-0.1,
        params={
            "free_force_change": 0.5,
            "free_moment_change": 0.01,
            "force_change_scale": 5.0,
            "moment_change_scale": 0.1,
            "max_positive_fraction": 0.05,
        },
    )
