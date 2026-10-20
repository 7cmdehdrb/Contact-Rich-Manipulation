"""Inherited clutter PPO with the live target position routed only to Critic."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter.ppo_cfg import ContactSweepClutterPPORunnerCfg, ObstacleEncoderModelCfg


@configclass
class PrivilegedObstacleEncoderModelCfg(ObstacleEncoderModelCfg):
    class_name = (
        "inspire_sweep_baseline.contact_sweep_clutter_v1.models:PrivilegedObstacleEncoderModel"
    )


@configclass
class ContactSweepClutterV1PPORunnerCfg(ContactSweepClutterPPORunnerCfg):
    experiment_name = "inspire_contact_sweep_clutter_v1"
    obs_groups = {
        "actor": ["policy", "obstacles"],
        "critic": ["policy", "privileged", "obstacles"],
    }
    critic = PrivilegedObstacleEncoderModelCfg()
