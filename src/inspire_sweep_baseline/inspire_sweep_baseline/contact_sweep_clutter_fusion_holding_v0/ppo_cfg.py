"""Independent experiment name; unchanged iteration-aware v1_0 runner."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.ppo_cfg import ContactSweepClutterFusionV1_0PPORunnerCfg


@configclass
class FusionHoldingV0PPORunnerCfg(ContactSweepClutterFusionV1_0PPORunnerCfg):
    experiment_name = 'inspire_contact_sweep_clutter_fusion_holding_v0'
