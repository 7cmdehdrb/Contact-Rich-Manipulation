"""Keep v1_1 PPO settings with a separate experiment name."""

from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_1.ppo_cfg import ContactSweepClutterFusionV1_1PPORunnerCfg


@configclass
class ContactSweepClutterFusionV1_1_1PPORunnerCfg(ContactSweepClutterFusionV1_1PPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_fusion_v1_1_1"
