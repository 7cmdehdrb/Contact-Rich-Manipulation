"""Keep the OSC v2 64→32→16 obstacle encoder for randomized props."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_osc_v2.ppo_cfg import FusionHoldingOSCV2PPORunnerCfg


@configclass
class FusionHoldingOSCObjectRandomV1PPORunnerCfg(FusionHoldingOSCV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v1"
