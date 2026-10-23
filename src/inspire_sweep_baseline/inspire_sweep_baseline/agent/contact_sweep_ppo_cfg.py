"""Contact Sweep PPO settings declared directly on the Isaac Lab runner base."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

@configclass
class ContactSweepPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = "inspire_contact_sweep"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.98, lam=0.95,
        desired_kl=0.02, max_grad_norm=1.0,
    )
