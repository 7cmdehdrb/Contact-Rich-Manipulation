"""Simple PPO configurations; network, optimizer and rollout settings preserved."""
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class SweepSimpleModelCfg(RslRlMLPModelCfg):
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
class SweepSimplePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_sweep_simple_v0'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']}
    policy = MISSING
    actor = SweepSimpleModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = SweepSimpleModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.98, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)

@configclass
class SweepSimpleProfPPORunnerCfg(SweepSimplePPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_prof_v0'

@configclass
class SweepSimpleV1PPORunnerCfg(SweepSimplePPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_v1'

@configclass
class SweepSimpleProfV1PPORunnerCfg(SweepSimpleProfPPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_prof_v1'


@configclass
class SweepSimpleHardPPORunnerCfg(SweepSimpleV1PPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_hard_v0'


@configclass
class SweepSimpleProfHardPPORunnerCfg(SweepSimpleProfV1PPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_prof_hard_v0'


@configclass
class SweepSimpleFingerModelCfg(SweepSimpleModelCfg):
    class_name = 'inspire_sweep_baseline.agent.sweep_simple_models:SweepSimpleFingerModel'


@configclass
class SweepSimpleFingerPPORunnerCfg(SweepSimpleV1PPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_finger_v0'
    actor = SweepSimpleFingerModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = SweepSimpleFingerModelCfg()


@configclass
class SweepSimpleFingerGaussianDistributionCfg:
    class_name: str = 'inspire_sweep_baseline.agent.sweep_simple_distributions:FingerGaussian'
    # None selects raw-unit stds: translation .2, rotation .5, hand range / 6.
    init_std: float | None = None


@configclass
class SweepSimpleFingerV1PPORunnerCfg(SweepSimpleFingerPPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_finger_v1'
    actor = SweepSimpleFingerModelCfg(distribution_cfg=SweepSimpleFingerGaussianDistributionCfg())


@configclass
class SweepSimpleFingerV2PPORunnerCfg(SweepSimpleFingerV1PPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_finger_v2'


@configclass
class SweepSimpleSemiProfPPORunnerCfg(SweepSimpleFingerV1PPORunnerCfg):
    experiment_name = 'inspire_sweep_simple_semi_prof_v0'
