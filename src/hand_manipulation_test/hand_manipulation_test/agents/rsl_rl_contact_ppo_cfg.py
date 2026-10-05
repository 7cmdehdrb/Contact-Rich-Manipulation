"""Inherited PPO defaults with an isolated Cube-contact experiment."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import HandManipulationTestPPORunnerCfg


@configclass
class ContactPPORunnerCfg(HandManipulationTestPPORunnerCfg):
    experiment_name = "hand_manipulation_contact"
