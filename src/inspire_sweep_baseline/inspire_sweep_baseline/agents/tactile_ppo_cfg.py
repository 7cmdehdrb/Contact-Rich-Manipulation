"""Baseline PPO parameters with independent tactile experiment storage."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import InspireSweepPPORunnerCfg


@configclass
class InspireSweepTactilePPORunnerCfg(InspireSweepPPORunnerCfg):
    experiment_name = "inspire_sweep_tactile"
