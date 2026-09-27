"""Sweep-Policy ``rsl_rl_ppo_cfg_02`` defaults adapted to this task.

The legacy local configuration remains available in :mod:`rsl_rl_ppo_cfg`.
This is the default used by task registration and the standalone scripts.
Only the requested 10,000-iteration limit differs from the reference.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BlindSweepReferencePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 36
    max_iterations = 10_000
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = None
    save_interval = 100
    experiment_name = "UR5e_shelf_sweep_random"
    run_name = ""
    logger = "tensorboard"

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


__all__ = ["BlindSweepReferencePPORunnerCfg"]
