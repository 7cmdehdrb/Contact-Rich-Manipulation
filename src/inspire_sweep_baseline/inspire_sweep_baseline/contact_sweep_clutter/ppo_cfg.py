"""Contact Sweep PPO with separately trained actor and critic obstacle encoders."""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ..contact_sweep.ppo_cfg import ContactSweepPPORunnerCfg


@configclass
class ObstacleEncoderModelCfg(RslRlMLPModelCfg):
    class_name = "inspire_sweep_baseline.contact_sweep_clutter.models:ObstacleEncoderModel"
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
