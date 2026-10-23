"""Independent runner/model Cfgs with the existing Clutter v1 PPO settings."""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ContactSweepClutterFusionModelCfg(RslRlMLPModelCfg):
    class_name = "inspire_sweep_baseline.contact_sweep_clutter_fusion.models:ContactSweepClutterFusionModel"
    hidden_dims = [256, 128, 64]
    activation = "elu"
    obs_normalization = False
    obstacle_group: str = "obstacles"
    encoder_hidden_dims: list[int] = [32]
    obstacle_observation_slots: int = 10
    embedding_dim: int = 16
    encoder_activation: str = "relu"


@configclass
class ContactSweepClutterFusionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "CurriculumOnPolicyRunner"
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = "inspire_contact_sweep_clutter_fusion"
    run_name = ""
    obs_groups = {
        "actor": ["policy", "obstacles"],
        "critic": ["policy", "privileged", "obstacles"],
    }
    policy = MISSING
    actor = ContactSweepClutterFusionModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = ContactSweepClutterFusionModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.98, lam=0.95,
        desired_kl=0.02, max_grad_norm=1.0,
    )
