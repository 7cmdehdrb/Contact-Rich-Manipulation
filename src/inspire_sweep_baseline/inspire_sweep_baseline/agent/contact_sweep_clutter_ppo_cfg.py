"""Contact Sweep PPO with separately trained actor and critic obstacle encoders.

Inherited clutter PPO with the live target position routed only to Critic.

Unchanged Clutter v1 PPO with an iteration-aware runner bridge.

Preserve v1 networks and route v2 training into a separate experiment."""

from .contact_sweep_ppo_cfg import ContactSweepPPORunnerCfg
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

@configclass
class ObstacleEncoderModelCfg(RslRlMLPModelCfg):
    class_name = 'inspire_sweep_baseline.agent.contact_sweep_clutter_models:ObstacleEncoderModel'
    hidden_dims = [256, 128, 64]
    activation = "elu"
    obs_normalization = False
    obstacle_group: str = "obstacles"
    encoder_hidden_dims: list[int] = [32]
    embedding_dim: int = 16
    encoder_activation: str = "relu"


@configclass
class ContactSweepClutterPPORunnerCfg(ContactSweepPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter"
    obs_groups = {
        "actor": ["policy", "obstacles"],
        "critic": ["policy", "obstacles"],
    }
    # Use the installed RSL-RL 5 model API; inherited PPO hyperparameters stay
    # unchanged. Each model owns and optimizes its own obstacle encoder.
    policy = MISSING
    actor = ObstacleEncoderModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = ObstacleEncoderModelCfg()


@configclass
class PrivilegedObstacleEncoderModelCfg(ObstacleEncoderModelCfg):
    class_name = (
        'inspire_sweep_baseline.agent.contact_sweep_clutter_models:PrivilegedObstacleEncoderModel'
    )


@configclass
class ContactSweepClutterV1PPORunnerCfg(ContactSweepClutterPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_v1"
    obs_groups = {
        "actor": ["policy", "obstacles"],
        "critic": ["policy", "privileged", "obstacles"],
    }
    critic = PrivilegedObstacleEncoderModelCfg()


@configclass
class ContactSweepClutterV1CurriculumPPORunnerCfg(ContactSweepClutterV1PPORunnerCfg):
    class_name = "CurriculumOnPolicyRunner"
    experiment_name = "inspire_contact_sweep_clutter_v1_curriculum"


@configclass
class ContactSweepClutterV2PPORunnerCfg(ContactSweepClutterV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_v2"
