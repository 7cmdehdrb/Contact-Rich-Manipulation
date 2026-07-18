"""Agent configurations for the OSC sweep task."""

from .rsl_rl_ppo_cfg import UR5eOscSweepPPORunnerCfg
from .rsl_rl_ppo_cfg_constant_velocity import (
    UR5eOscSweepConstantVelocityPPORunnerCfg,
)

__all__ = [
    "UR5eOscSweepPPORunnerCfg",
    "UR5eOscSweepConstantVelocityPPORunnerCfg",
]
