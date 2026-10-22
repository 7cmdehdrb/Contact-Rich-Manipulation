"""Inherit PPO unchanged; separate the experiment directory."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.ppo_cfg import ContactSweepClutterFusionV1_1PPORunnerCfg


@configclass
class ContactSweepClutterFusionV1_2PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_2"
