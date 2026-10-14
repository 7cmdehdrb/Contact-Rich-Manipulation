"""Learning configurations for the independent OSC sweep task."""

from .rsl_rl_ppo_cfg import UR5eOscSweepIndependentPPORunnerCfg
from .rsl_rl_ppo_cfg_detailed import UR5eOscSweepIndependentDetailedPPORunnerCfg

__all__ = [
    "UR5eOscSweepIndependentDetailedPPORunnerCfg",
    "UR5eOscSweepIndependentPPORunnerCfg",
]
