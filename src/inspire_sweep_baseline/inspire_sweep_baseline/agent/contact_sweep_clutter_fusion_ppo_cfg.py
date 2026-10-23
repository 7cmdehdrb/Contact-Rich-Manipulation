"""Independent runner/model Cfgs with the existing Clutter v1 PPO settings.

Inherit Fusion v0 PPO settings with a separate experiment directory.

Preserve Fusion PPO and iteration runner with a separate experiment name.

Preserve the iteration-aware Fusion runner under a separate experiment.

Inherit PPO unchanged; separate the experiment directory.

Keep v1_1 PPO settings with a separate experiment name.

Keep v1_3 PPO settings with a separate experiment name.

Keep v1.0 policy/checkpoint compatibility under a separate experiment."""

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

@configclass
class ContactSweepClutterFusionModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_models:ContactSweepClutterFusionModel'
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


@configclass
class ContactSweepClutterFusionV2PPORunnerCfg(ContactSweepClutterFusionPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v2"


@configclass
class ContactSweepClutterFusionV3PPORunnerCfg(ContactSweepClutterFusionV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v3"


@configclass
class ContactSweepClutterFusionV1_0PPORunnerCfg(ContactSweepClutterFusionV3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_0"


@configclass
class ContactSweepClutterFusionV1_1PPORunnerCfg(ContactSweepClutterFusionV1_0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_1"


@configclass
class ContactSweepClutterFusionV1_1_1PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_1_1"


@configclass
class ContactSweepClutterFusionV1_1_2PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_1_2"


@configclass
class ContactSweepClutterFusionV1_2PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_2"


@configclass
class ContactSweepClutterFusionV1_3PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_3"


@configclass
class ContactSweepClutterFusionV1_3_1PPORunnerCfg(ContactSweepClutterFusionV1_3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_3_1"


@configclass
class ContactSweepClutterFusionV1_4PPORunnerCfg(ContactSweepClutterFusionV1_0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_4"
