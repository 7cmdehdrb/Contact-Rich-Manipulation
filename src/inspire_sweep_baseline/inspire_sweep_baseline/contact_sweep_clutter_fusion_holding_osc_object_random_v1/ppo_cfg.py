"""Keep the OSC v1 64→32→16 obstacle encoder for randomized props."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_osc_v1.ppo_cfg import FusionHoldingOSCV1PPORunnerCfg


@configclass
class FusionHoldingOSCObjectRandomV1PPORunnerCfg(FusionHoldingOSCV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v1"
