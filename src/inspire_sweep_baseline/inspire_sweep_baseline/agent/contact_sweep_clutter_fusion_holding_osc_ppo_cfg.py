"""Unchanged Holding PPO with an independent experiment name.

Holding v1 encoder and PPO with an independent OSC experiment name.

Keep Holding v2 PPO and encoder under a separate experiment name."""

from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV0PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV1PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV2PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV3PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV4PPORunnerCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingOSCV0PPORunnerCfg(FusionHoldingV0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v0"


@configclass
class FusionHoldingOSCV1PPORunnerCfg(FusionHoldingV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v1"


@configclass
class FusionHoldingOSCV2PPORunnerCfg(FusionHoldingV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v2"


@configclass
class FusionHoldingOSCV3PPORunnerCfg(FusionHoldingV3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v3"


@configclass
class FusionHoldingOSCV4PPORunnerCfg(FusionHoldingV4PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v4"
