"""Inherit Fusion v0 PPO settings with a separate experiment directory."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.ppo_cfg import ContactSweepClutterFusionPPORunnerCfg


@configclass
class ContactSweepClutterFusionV2PPORunnerCfg(ContactSweepClutterFusionPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v2"
