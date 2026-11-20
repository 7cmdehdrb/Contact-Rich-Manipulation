from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class StabilizeSweepModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.stabilize_sweep_models:StabilizeSweepModel'
    hidden_dims = [256, 128, 64]
    activation = 'elu'
    obs_normalization = False
    obstacle_group: str = 'obstacles'
    encoder_hidden_dims: list[int] = [32]
    obstacle_observation_slots: int = 16
    embedding_dim: int = 16
    encoder_activation: str = 'relu'

@configclass
class StabilizeSweepPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'CurriculumOnPolicyRunner'
    # Execution-wrapper bound. The Gaussian actor and PPO storage remain unbounded.
    clip_actions = 1.5707963267948966
    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = 'inspire_stabilize_sweep_v0'
    run_name = ''
    obs_groups = {'actor': ['policy', 'obstacles'], 'critic': ['policy', 'obstacles']}
    policy = MISSING
    actor = StabilizeSweepModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = StabilizeSweepModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8, num_mini_batches=4, learning_rate=0.001, schedule='adaptive', gamma=0.98, lam=0.95, desired_kl=0.02, max_grad_norm=1.0)


@configclass
class SweepOnlyModelCfg(StabilizeSweepModelCfg):
    class_name = 'inspire_sweep_baseline.agent.stabilize_sweep_models:SweepOnlyModel'


@configclass
class SweepOnlyPPORunnerCfg(StabilizeSweepPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v0'
    actor = SweepOnlyModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = SweepOnlyModelCfg()


@configclass
class StabilizeSweepTestPPORunnerCfg(StabilizeSweepPPORunnerCfg):
    experiment_name = 'inspire_stabilize_sweep_v0_test'


@configclass
class SweepOnlyTestPPORunnerCfg(SweepOnlyPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v0_test'


@configclass
class SweepOnlyV1TestPPORunnerCfg(SweepOnlyTestPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v1_test'


@configclass
class SweepOnlyV1_2TestPPORunnerCfg(SweepOnlyV1TestPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v1_2_test'


@configclass
class SweepOnlyV2TestPPORunnerCfg(SweepOnlyV1TestPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v2_test'


@configclass
class SweepOnlyV2_2TestPPORunnerCfg(SweepOnlyV1_2TestPPORunnerCfg):
    experiment_name = 'inspire_sweep_only_v2_2_test'


@configclass
class SweepOnlyActionPenaltyPPORunnerCfg(SweepOnlyPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v0_action_penalty'


@configclass
class SweepOnlyTestActionPenaltyPPORunnerCfg(SweepOnlyTestPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v0_test_action_penalty'


@configclass
class SweepOnlyV1TestActionPenaltyPPORunnerCfg(SweepOnlyV1TestPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v1_test_action_penalty'


@configclass
class SweepOnlyV1_2TestActionPenaltyPPORunnerCfg(SweepOnlyV1_2TestPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v1_2_test_action_penalty'


@configclass
class SweepOnlyV2TestActionPenaltyPPORunnerCfg(SweepOnlyV2TestPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v2_test_action_penalty'


@configclass
class SweepOnlyV2_2TestActionPenaltyPPORunnerCfg(SweepOnlyV2_2TestPPORunnerCfg):
    # Bounds live in action terms so rewards can inspect the original command.
    clip_actions = None
    experiment_name = 'inspire_sweep_only_v2_2_test_action_penalty'
