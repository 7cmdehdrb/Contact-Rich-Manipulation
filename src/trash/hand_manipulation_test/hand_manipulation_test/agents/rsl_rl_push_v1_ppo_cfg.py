"""Inherited Sweep-Policy cfg_02 PPO defaults with separate Push v1 logs."""

from isaaclab.utils import configclass

from .rsl_rl_push_ppo_cfg import PushPPORunnerCfg


@configclass
class PushV1PPORunnerCfg(PushPPORunnerCfg):
    experiment_name = "hand_manipulation_push_v1"
