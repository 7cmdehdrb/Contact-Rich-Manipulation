"""Inherit PPO unchanged; separate the experiment directory."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.ppo_cfg import ContactSweepClutterFusionV1_0PPORunnerCfg


@configclass
class ContactSweepClutterFusionV1_1PPORunnerCfg(ContactSweepClutterFusionV1_0PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_1"
