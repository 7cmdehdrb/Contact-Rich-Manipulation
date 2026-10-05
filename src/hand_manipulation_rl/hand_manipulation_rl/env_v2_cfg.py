"""Reward-coefficient configuration for the inherited Approach-v2 task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .env_v1_cfg import BlindSweepApproachV1EnvCfg, BlindSweepApproachV1RewardsCfg
from .mdp.rewards import (
    actual_contact_normal_alignment,
    object_goal_reward,
    success_terminal_bonus_rate,
)


@configclass
class BlindSweepApproachV2RewardsCfg(BlindSweepApproachV1RewardsCfg):
    """Prioritize pushing to the goal while preserving approach exploration."""

    # v1 training around iteration 1900 produced contact and ~33 mm object
    # motion, but its weak goal gradient did not make continued pushing the
    # dominant behavior.  Keep approach/contact shaping unchanged and rebalance
    # only the goal-facing terms.
    goal = RewTerm(func=object_goal_reward, weight=12.0)
    actual_normal_alignment = RewTerm(
        func=actual_contact_normal_alignment,
        weight=4.0,
    )
    # The dense goal term is an occupancy reward.  A larger terminal bonus
    # keeps crossing the success threshold preferable to lingering near it.
    success_terminal = RewTerm(func=success_terminal_bonus_rate, weight=4.0)


@configclass
class BlindSweepApproachV2EnvCfg(BlindSweepApproachV1EnvCfg):
    """Approach-v1 inherited unchanged except for reward coefficients."""

    rewards: BlindSweepApproachV2RewardsCfg = BlindSweepApproachV2RewardsCfg()


__all__ = ["BlindSweepApproachV2EnvCfg", "BlindSweepApproachV2RewardsCfg"]
