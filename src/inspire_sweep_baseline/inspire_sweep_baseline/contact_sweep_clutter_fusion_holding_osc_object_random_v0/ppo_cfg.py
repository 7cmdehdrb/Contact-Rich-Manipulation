from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_holding_osc_v0.ppo_cfg import FusionHoldingOSCV0PPORunnerCfg


@configclass
class FusionHoldingOSCObjectRandomV0PPORunnerCfg(FusionHoldingOSCV0PPORunnerCfg):
    experiment_name = 'inspire_contact_sweep_clutter_fusion_holding_osc_object_random_v0'

    def __post_init__(self):
        self.actor.obstacle_observation_slots = 14
        self.critic.obstacle_observation_slots = 14
