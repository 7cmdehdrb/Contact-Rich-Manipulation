"""OmniV2 mechanics with DexTouch-style distance-record reward shaping."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .mdp import dex_rewards
from .omni_env_cfg import InspireSweepOmniV2EnvCfg, OmniV2RewardsCfg


@configclass
class DexOmniRewardsCfg(OmniV2RewardsCfg):
    # Experimental sweep coefficients in reward/meter, not paper parameters or
    # normalized episode budgets. A 14 cm goal improvement pays 42 reward.
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
class InspireDexOmniEnvCfg(InspireSweepOmniV2EnvCfg):
    """Change only task shaping; inherit penalties, termination and spawning."""

    rewards: DexOmniRewardsCfg = DexOmniRewardsCfg()
