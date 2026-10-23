"""Holding PPO with independent 64 -> 32 -> 16 actor and critic encoders."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v0.ppo_cfg import FusionHoldingV0PPORunnerCfg


@configclass
class FusionHoldingV1PPORunnerCfg(FusionHoldingV0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_v1"

    def __post_init__(self):
        for model in (self.actor, self.critic):
            model.obstacle_observation_slots = 16
            model.encoder_hidden_dims = [32]
            model.embedding_dim = 16
