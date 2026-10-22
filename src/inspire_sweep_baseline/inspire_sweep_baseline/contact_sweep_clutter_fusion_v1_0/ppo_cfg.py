"""Preserve the iteration-aware Fusion runner under a separate experiment."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v3.ppo_cfg import ContactSweepClutterFusionV3PPORunnerCfg


@configclass
class ContactSweepClutterFusionV1_0PPORunnerCfg(ContactSweepClutterFusionV3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_0"
