"""RSL-RL logging namespace for the inherited approach-shaped task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_02 import BlindSweepReferencePPORunnerCfg


@configclass
class BlindSweepApproachPPORunnerCfg(BlindSweepReferencePPORunnerCfg):
    """Reuse the 57-D/8-D PPO architecture while separating run outputs."""

    experiment_name = "UR5e_shelf_sweep_approach"


__all__ = ["BlindSweepApproachPPORunnerCfg"]
