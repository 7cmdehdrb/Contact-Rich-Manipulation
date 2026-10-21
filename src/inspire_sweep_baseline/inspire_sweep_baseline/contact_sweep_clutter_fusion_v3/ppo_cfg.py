"""Preserve Fusion PPO and iteration runner with a separate experiment name."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v2.ppo_cfg import ContactSweepClutterFusionV2PPORunnerCfg


@configclass
class ContactSweepClutterFusionV3PPORunnerCfg(ContactSweepClutterFusionV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v3"
