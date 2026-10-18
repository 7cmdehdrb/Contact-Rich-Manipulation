"""Dex Tactile v1 with goal occupancy rewards continuing until timeout."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .dex_omni_renewal_env_cfg import DexOmniRenewalRewardsCfg
from .dex_tactile_omni_renewal_v1_env_cfg import InspireDexTactileSweepOmniRenewalV1EnvCfg
from .mdp import goal_rewards
from .mdp.goal_hold_rewards import StableGoalAchievement
from .omni_env_cfg import OBJECT_RADIUS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT
from .omni_renewal_env_cfg import OmniRenewalTerminationsCfg


# Only active failures inherited from v1; object_outside_shelf is disabled.
FAILURE_TERMS = (
    "object_drop", "push_fast", "shelf_collision", "hand_velocity", "shelf_displacement",
)


@configclass
class DexTactileOmniRenewalV2RewardsCfg(DexOmniRenewalRewardsCfg):
    # Evaluate the shared timer first; latch +100 without resetting the env.
    success = RewTerm(
        func=StableGoalAchievement,
        weight=100.0,
        params={
            "position_tolerance": 0.01,
            "speed_tolerance": 0.02,
            "upright_angle": math.radians(10.0),
            "hold_time": 0.5,
            "surface_height": SHELF_SURFACE_HEIGHT,
            "support_tolerance": 0.01,
            "half_extents": SHELF_HALF_EXTENTS,
            "object_radius": OBJECT_RADIUS,
            "failure_terms": FAILURE_TERMS,
        },
    )
    goal_hold = RewTerm(func=goal_rewards.stable_goal_hold, weight=20.0)


@configclass
class DexTactileOmniRenewalV2TerminationsCfg(OmniRenewalTerminationsCfg):
    # Achievement is a reward event. Only inherited failures/timeout end runs.
    success = None


@configclass
class InspireDexTactileSweepOmniRenewalV2EnvCfg(InspireDexTactileSweepOmniRenewalV1EnvCfg):
    """Inherit v1 mechanics and reward stable occupancy until failure/timeout."""

    rewards: DexTactileOmniRenewalV2RewardsCfg = DexTactileOmniRenewalV2RewardsCfg()
    terminations: DexTactileOmniRenewalV2TerminationsCfg = DexTactileOmniRenewalV2TerminationsCfg()
