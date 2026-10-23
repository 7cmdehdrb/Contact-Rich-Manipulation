"""Independent experiment name; unchanged iteration-aware v1_0 runner.

Holding PPO with independent 64 -> 32 -> 16 actor and critic encoders.

Keep Holding v1 PPO and encoders under a separate experiment name."""

from .contact_sweep_clutter_fusion_ppo_cfg import ContactSweepClutterFusionV1_0PPORunnerCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingV0PPORunnerCfg(ContactSweepClutterFusionV1_0PPORunnerCfg):
    experiment_name = 'inspire_contact_sweep_clutter_fusion_holding_v0'


@configclass
class FusionHoldingV1PPORunnerCfg(FusionHoldingV0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_v1"

    def __post_init__(self):
        for model in (self.actor, self.critic):
            model.obstacle_observation_slots = 16
            model.encoder_hidden_dims = [32]
            model.embedding_dim = 16


@configclass
class FusionHoldingV2PPORunnerCfg(FusionHoldingV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_v2"
