"""Inherited PPO unchanged; use a separate experiment directory."""

from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import InspireSweepPPORunnerCfg


@configclass
class InspireSweepRealisticPhysicsPPORunnerCfg(InspireSweepPPORunnerCfg):
    experiment_name = "inspire_sweep_realistic_physics"
