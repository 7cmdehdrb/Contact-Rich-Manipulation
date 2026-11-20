"""Sweep v0 PPO; independent runner declaration with the reference hyperparameters."""
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from .stabilize_sweep_ppo_cfg import SweepOnlyModelCfg


@configclass
class SweepPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    # Raw actions reach the manager; execution bounds remain in Action terms.
    clip_actions = None
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_sweep_v0'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']}
    policy = MISSING
    actor = SweepOnlyModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = SweepOnlyModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.98, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)

