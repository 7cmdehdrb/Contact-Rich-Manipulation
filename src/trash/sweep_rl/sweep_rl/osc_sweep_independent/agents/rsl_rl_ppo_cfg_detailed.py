"""PPO logging namespace for the detailed-reward independent task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UR5eOscSweepIndependentPPORunnerCfg


@configclass
class UR5eOscSweepIndependentDetailedPPORunnerCfg(
    UR5eOscSweepIndependentPPORunnerCfg
):
    experiment_name = "ur5e_osc_sweep_independent_detailed"
