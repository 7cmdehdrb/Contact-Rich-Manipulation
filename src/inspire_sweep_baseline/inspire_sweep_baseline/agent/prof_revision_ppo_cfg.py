"""Sweep v0 PPO; independent runner declaration with the reference hyperparameters."""
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
@configclass
class ProfRevisionModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.prof_revision_models:ProfRevisionModel'
    hidden_dims = [256, 128, 64]
    activation = 'elu'
    obs_normalization = False
    obstacle_group: str = 'obstacles'
    encoder_hidden_dims: list[int] = [128, 64, 32]
    obstacle_observation_slots: int = 16
    embedding_dim: int = 16
    encoder_activation: str = 'relu'



@configclass
class ProfRevisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    # Raw actions reach the manager; execution bounds remain in Action terms.
    clip_actions = None
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_sweep_prof_revision_v0'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']}
    policy = MISSING
    actor = ProfRevisionModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = ProfRevisionModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.98, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)


@configclass
class ProfRevisionReach5cmPPORunnerCfg(ProfRevisionPPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_reach5cm_v0'


@configclass
class ProfRevisionV1PPORunnerCfg(ProfRevisionPPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v1'


@configclass
class ProfRevisionV0_1PPORunnerCfg(ProfRevisionPPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_1'


@configclass
class ProfRevisionV0_2PPORunnerCfg(ProfRevisionPPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_2'


@configclass
class ProfRevisionV0_3PPORunnerCfg(ProfRevisionPPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3'


@configclass
class ProfRevisionHardV0_3PPORunnerCfg(ProfRevisionV0_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_hard_v0_3'


@configclass
class ProfRevisionV0_3_1PPORunnerCfg(ProfRevisionV0_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_1'


@configclass
class ProfRevisionV0_3_2PPORunnerCfg(ProfRevisionV0_3_1PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_2'


@configclass
class ProfRevisionV0_3_3PPORunnerCfg(ProfRevisionV0_3_2PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_3'


@configclass
class ProfRevisionV0_3_4PPORunnerCfg(ProfRevisionV0_3_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_4'


@configclass
class ProfRevisionHardV0_3_3PPORunnerCfg(ProfRevisionV0_3_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_hard_v0_3_3'


@configclass
class ProfRevisionV0_3_3IKPPORunnerCfg(ProfRevisionV0_3_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_3_ik'


@configclass
class ProfRevisionV0_3_3IKV1PPORunnerCfg(ProfRevisionV0_3_3PPORunnerCfg):
    experiment_name = 'inspire_sweep_prof_revision_v0_3_3_ik_v1'
