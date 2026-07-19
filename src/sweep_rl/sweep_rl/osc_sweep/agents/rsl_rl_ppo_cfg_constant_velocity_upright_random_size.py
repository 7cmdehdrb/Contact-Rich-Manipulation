"""PPO config for the upright, random-size constant-velocity task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_constant_velocity import (
    UR5eOscSweepConstantVelocityPPORunnerCfg,
)


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizePPORunnerCfg(
    UR5eOscSweepConstantVelocityPPORunnerCfg
):
    experiment_name = "ur5e_osc_sweep_constant_velocity_upright_random_size"
