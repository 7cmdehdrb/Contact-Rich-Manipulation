"""Keep Holding v2 PPO and encoder under a separate experiment name."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_holding_v2.ppo_cfg import FusionHoldingV2PPORunnerCfg


@configclass
class FusionHoldingOSCV2PPORunnerCfg(FusionHoldingV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v2"
