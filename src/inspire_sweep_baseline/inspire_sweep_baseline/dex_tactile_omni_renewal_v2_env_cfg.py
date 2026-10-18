"""Dex Tactile v0 with a stable-goal hold reward and one terminal bonus."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .dex_omni_renewal_env_cfg import DexOmniRenewalRewardsCfg
from .mdp import goal_rewards, goal_terminations
from .omni_env_cfg import OBJECT_RADIUS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT
from .omni_renewal_env_cfg import OmniRenewalTerminationsCfg
from .tactile_omni_renewal_env_cfg import InspireDexTactileSweepOmniRenewalEnvCfg


# Only active failures inherited from v0; object_outside_shelf is disabled.
FAILURE_TERMS = (
    "object_drop", "push_fast", "shelf_collision", "hand_velocity", "shelf_displacement",
)


@configclass
class DexTactileOmniRenewalV2RewardsCfg(DexOmniRenewalRewardsCfg):
    goal_hold = RewTerm(func=goal_rewards.stable_goal_hold, weight=20.0)
    # success_event cancels dt: +100 once, on the successful terminal step.
    success = RewTerm(func=goal_rewards.success_event, weight=100.0)


@configclass
class DexTactileOmniRenewalV2TerminationsCfg(OmniRenewalTerminationsCfg):
    # Keep this last: all failure flags must be computed before eligibility.
    success = DoneTerm(
        func=goal_terminations.StableGoalSuccess,
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


@configclass
class InspireDexTactileSweepOmniRenewalV2EnvCfg(InspireDexTactileSweepOmniRenewalEnvCfg):
    """Direct v0 descendant; add only positive rewards and stable success."""

    rewards: DexTactileOmniRenewalV2RewardsCfg = DexTactileOmniRenewalV2RewardsCfg()
    terminations: DexTactileOmniRenewalV2TerminationsCfg = DexTactileOmniRenewalV2TerminationsCfg()
