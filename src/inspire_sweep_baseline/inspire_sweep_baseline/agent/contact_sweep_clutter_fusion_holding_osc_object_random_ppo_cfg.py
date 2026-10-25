"""Holding v2 PPO with a separate joint-position randomization experiment.

Keep the OSC v2 64→32→16 obstacle encoder for randomized props."""

from .contact_sweep_clutter_fusion_holding_osc_ppo_cfg import FusionHoldingOSCV0PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_osc_ppo_cfg import FusionHoldingOSCV2PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_osc_ppo_cfg import FusionHoldingOSCV3PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_osc_ppo_cfg import FusionHoldingOSCV4PPORunnerCfg
from .contact_sweep_clutter_fusion_holding_ppo_cfg import FusionHoldingV2PPORunnerCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingObjectRandomTestV0PPORunnerCfg(FusionHoldingV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_object_random_test_v0"


@configclass
class FusionHoldingOSCObjectRandomV0PPORunnerCfg(FusionHoldingOSCV0PPORunnerCfg):
    experiment_name = 'inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v0'

    def __post_init__(self):
        self.actor.obstacle_observation_slots = 14
        self.critic.obstacle_observation_slots = 14


@configclass
class FusionHoldingOSCObjectRandomV1PPORunnerCfg(FusionHoldingOSCV2PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v1"


@configclass
class FusionHoldingOSCObjectRandomV2PPORunnerCfg(FusionHoldingOSCV3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v2"


@configclass
class FusionHoldingOSCObjectRandomV3PPORunnerCfg(FusionHoldingOSCV4PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v3"
