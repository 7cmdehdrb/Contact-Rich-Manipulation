"""Unchanged Holding PPO with an independent experiment name."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_holding_v0.ppo_cfg import FusionHoldingV0PPORunnerCfg


@configclass
class FusionHoldingOSCV0PPORunnerCfg(FusionHoldingV0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v0"
