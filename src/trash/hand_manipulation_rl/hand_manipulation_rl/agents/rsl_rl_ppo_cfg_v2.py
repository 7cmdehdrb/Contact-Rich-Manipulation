"""PPO runner namespace for the reward-rebalanced Approach-v2 task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_v1 import BlindSweepApproachV1PPORunnerCfg


@configclass
class BlindSweepApproachV2PPORunnerCfg(BlindSweepApproachV1PPORunnerCfg):
    """Reuse v1 PPO hyperparameters while separating logs/checkpoints."""

    experiment_name = "UR5e_shelf_sweep_approach_v2"


__all__ = ["BlindSweepApproachV2PPORunnerCfg"]
