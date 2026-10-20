"""Unchanged Clutter v1 PPO with an iteration-aware runner bridge."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.ppo_cfg import ContactSweepClutterV1PPORunnerCfg


@configclass
class ContactSweepClutterV1CurriculumPPORunnerCfg(ContactSweepClutterV1PPORunnerCfg):
    class_name = "CurriculumOnPolicyRunner"
    experiment_name = "inspire_contact_sweep_clutter_v1_curriculum"
