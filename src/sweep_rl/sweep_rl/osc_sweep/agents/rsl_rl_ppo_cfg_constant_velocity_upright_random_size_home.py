"""PPO config for sweep followed by collision-free Home return."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_constant_velocity_upright_random_size import (
    UR5eOscSweepConstantVelocityUprightRandomSizePPORunnerCfg,
)


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizeHomePPORunnerCfg(
    UR5eOscSweepConstantVelocityUprightRandomSizePPORunnerCfg
):
    experiment_name = "ur5e_osc_sweep_constant_velocity_upright_random_size_home"
