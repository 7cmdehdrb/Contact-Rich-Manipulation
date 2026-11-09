"""Independent PPO configuration matching symmetric IK-v0_1."""
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class WrenchRefactorRelativeIKV0_2TestModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.wrench_refactor_models:WrenchRefactorRelativeModel'
    hidden_dims = [256, 128, 64]
    activation = 'elu'
    obs_normalization = False
    obstacle_group: str = 'obstacles'
    encoder_hidden_dims: list[int] = [32]
    obstacle_observation_slots: int = 16
    embedding_dim: int = 16
    encoder_activation: str = 'relu'

@configclass
class WrenchRefactorRelativeIKV0_2TestPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2-Test'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']}
    policy = MISSING
    actor = WrenchRefactorRelativeIKV0_2TestModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = WrenchRefactorRelativeIKV0_2TestModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.98, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)


@configclass
class WrenchRefactorRelativeIKV0_2_1TestPPORunnerCfg(WrenchRefactorRelativeIKV0_2TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_1-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_1ResetTestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_1TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_1-Reset-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_2TestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_1TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_2-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_2_1TestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_2TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_2_1-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_2_2TestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_2_1TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_2_2-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_2_2FastTestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_2_2TestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_2_2-Fast-Test'


@configclass
class WrenchRefactorRelativeIKV0_2_2_3FastTestPPORunnerCfg(WrenchRefactorRelativeIKV0_2_2_2FastTestPPORunnerCfg):
    experiment_name = 'inspire_wrench_refactor_relative_ik_v0_2_2_3-Fast-Test'
