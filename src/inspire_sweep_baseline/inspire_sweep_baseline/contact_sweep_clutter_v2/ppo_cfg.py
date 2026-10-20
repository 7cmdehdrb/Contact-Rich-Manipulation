"""Preserve v1 networks and route v2 training into a separate experiment."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.ppo_cfg import ContactSweepClutterV1PPORunnerCfg


@configclass
class ContactSweepClutterV2PPORunnerCfg(ContactSweepClutterV1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_v2"
