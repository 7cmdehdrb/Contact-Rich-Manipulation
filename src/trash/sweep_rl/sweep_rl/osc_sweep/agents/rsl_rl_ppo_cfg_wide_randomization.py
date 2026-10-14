"""PPO configuration copy for the wide-randomization sweep experiment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UR5eOscSweepPPORunnerCfg


@configclass
class UR5eOscSweepWideRandomizationPPORunnerCfg(UR5eOscSweepPPORunnerCfg):
    """Keep the baseline PPO settings but isolate checkpoints and logs."""

    experiment_name = "ur5e_osc_sweep_wide_randomization"

