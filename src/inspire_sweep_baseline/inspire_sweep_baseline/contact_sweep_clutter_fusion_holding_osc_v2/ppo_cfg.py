"""Keep Holding v1 PPO and encoder under a separate experiment name."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_holding_v1.ppo_cfg import FusionHoldingV1PPORunnerCfg


@configclass
class FusionHoldingOSCV2PPORunnerCfg(FusionHoldingV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_v2"
