"""Stable PPO defaults for the corrected Approach-v1 task."""

from copy import deepcopy

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_approach import BlindSweepApproachPPORunnerCfg


@configclass
class BlindSweepApproachV1PPORunnerCfg(BlindSweepApproachPPORunnerCfg):
    """Lower initial exploration and prevent adaptive LR escalation."""

    experiment_name = "UR5e_shelf_sweep_approach_v1"

    actor = deepcopy(BlindSweepApproachPPORunnerCfg().actor)
    actor.distribution_cfg.init_std = 0.5

    algorithm = deepcopy(BlindSweepApproachPPORunnerCfg().algorithm)
    algorithm.learning_rate = 3.0e-4
    algorithm.schedule = "fixed"


__all__ = ["BlindSweepApproachV1PPORunnerCfg"]
