"""PPO configuration copy for tactile-only target-state estimation."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_wide_randomization import UR5eOscSweepWideRandomizationPPORunnerCfg


@configclass
class UR5eOscSweepTactileLocalizationPPORunnerCfg(UR5eOscSweepWideRandomizationPPORunnerCfg):
    """Use a separate experiment directory for the 56-D policy."""

    experiment_name = "ur5e_osc_sweep_tactile_localization"

