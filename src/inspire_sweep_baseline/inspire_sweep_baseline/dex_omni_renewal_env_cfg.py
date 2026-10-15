"""Renewal mechanics with only the two Dex distance-progress rewards replaced."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .mdp import dex_rewards
from .omni_renewal_env_cfg import InspireSweepOmniRenewalEnvCfg, OmniRenewalRewardsCfg


@configclass
class DexOmniRenewalRewardsCfg(OmniRenewalRewardsCfg):
    reaching = RewTerm(
        func=dex_rewards.DexReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=dex_rewards.DexPlanarGoalProgress,
        weight=300.0,
        params={"position_tolerance": 0.01},
    )


@configclass
class InspireDexSweepOmniRenewalEnvCfg(InspireSweepOmniRenewalEnvCfg):
    rewards: DexOmniRenewalRewardsCfg = DexOmniRenewalRewardsCfg()
