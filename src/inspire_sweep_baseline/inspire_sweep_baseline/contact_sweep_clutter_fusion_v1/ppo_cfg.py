"""Preserve Fusion PPO and iteration runner with a separate experiment name."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion.ppo_cfg import ContactSweepClutterFusionPPORunnerCfg


@configclass
class ContactSweepClutterFusionV1PPORunnerCfg(ContactSweepClutterFusionPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1"
