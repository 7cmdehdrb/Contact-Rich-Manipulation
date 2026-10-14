"""Reward-rebalanced environment inheriting the complete Approach-v1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .env_v1 import BlindSweepApproachV1Env

if TYPE_CHECKING:
    from .env_v2_cfg import BlindSweepApproachV2EnvCfg


class BlindSweepApproachV2Env(BlindSweepApproachV1Env):
    """Approach-v1 dynamics with push-prioritized reward coefficients."""

    cfg: "BlindSweepApproachV2EnvCfg"


__all__ = ["BlindSweepApproachV2Env"]
