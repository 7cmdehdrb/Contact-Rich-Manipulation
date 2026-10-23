"""Keep Holding v1 PPO and encoders under a separate experiment name."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v1.ppo_cfg import FusionHoldingV1PPORunnerCfg


@configclass
class FusionHoldingV2PPORunnerCfg(FusionHoldingV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_v2"
