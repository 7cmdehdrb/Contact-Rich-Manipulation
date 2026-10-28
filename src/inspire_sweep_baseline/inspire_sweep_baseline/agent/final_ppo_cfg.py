"""Independent Final v0 PPO configuration with the existing model and runner."""

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class FinalModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_models:ContactSweepClutterFusionModel'
    hidden_dims = [256, 128, 64]
    activation = 'elu'
    obs_normalization = False
    obstacle_group: str = 'obstacles'
    encoder_hidden_dims: list[int] = [32]
    obstacle_observation_slots: int = 16
    embedding_dim: int = 16
    encoder_activation: str = 'relu'


@configclass
class FinalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_final_v0'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'privileged', 'obstacles']}
    policy = MISSING
    actor = FinalModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = FinalModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule='adaptive',
        gamma=0.98,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class FinalV1ModelCfg(FinalModelCfg):
    class_name = 'inspire_sweep_baseline.agent.final_models:FinalV1Model'
    obstacle_observation_slots: int = 20


@configclass
class FinalV1PPORunnerCfg(FinalPPORunnerCfg):
    experiment_name = 'inspire_final_v1'
    actor = FinalV1ModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = FinalV1ModelCfg()


@configclass
class FinalV1_1PPORunnerCfg(FinalV1PPORunnerCfg):
    experiment_name = 'inspire_final_v1_1'


@configclass
class FinalV1TestPPORunnerCfg(FinalV1PPORunnerCfg):
    experiment_name = 'inspire_final_v1_test'


@configclass
class FinalGoalAccuracyPPORunnerCfg(FinalV1PPORunnerCfg):
    experiment_name = 'inspire_final_goal_accuracy_v0'


@configclass
class FinalGoalAccuracyV1PPORunnerCfg(FinalGoalAccuracyPPORunnerCfg):
    experiment_name = 'inspire_final_goal_accuracy_v1'


@configclass
class FinalGoalAccuracyV0_1PPORunnerCfg(FinalGoalAccuracyPPORunnerCfg):
    experiment_name = 'inspire_final_goal_accuracy_v0_1'


@configclass
class FinalV2PPORunnerCfg(FinalV1PPORunnerCfg):
    experiment_name = 'inspire_final_v2'
