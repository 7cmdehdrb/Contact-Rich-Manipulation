"""Keep v1_3 PPO settings with a separate experiment name."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_3.ppo_cfg import ContactSweepClutterFusionV1_3PPORunnerCfg


@configclass
class ContactSweepClutterFusionV1_3_1PPORunnerCfg(ContactSweepClutterFusionV1_3PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_3_1"
