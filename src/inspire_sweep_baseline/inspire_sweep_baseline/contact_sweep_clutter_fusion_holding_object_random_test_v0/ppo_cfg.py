"""Holding v2 PPO with a separate joint-position randomization experiment."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v2.ppo_cfg import FusionHoldingV2PPORunnerCfg


@configclass
class FusionHoldingObjectRandomTestV0PPORunnerCfg(FusionHoldingV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_object_random_test_v0"
