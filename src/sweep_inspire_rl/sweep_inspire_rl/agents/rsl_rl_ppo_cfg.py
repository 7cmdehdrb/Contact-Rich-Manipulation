"""Preserve every Sweep-Policy PPO setting while naming the new experiment."""

from isaaclab.utils import configclass

from sweeping_policy.config.ur5e.agents.rsl_rl_ppo_cfg_02 import UR5eSweepPPORunnerCfg


@configclass
class InspireShelfSweepPPORunnerCfg(UR5eSweepPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_inspire_right"


@configclass
class InspireShelfSweepV1PPORunnerCfg(InspireShelfSweepPPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_inspire_right_v1"


@configclass
class InspireShelfSweepV2PPORunnerCfg(InspireShelfSweepV1PPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_inspire_right_v2"


@configclass
class InspireShelfSweepV3PPORunnerCfg(InspireShelfSweepV1PPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_inspire_right_v3"


@configclass
class InspireShelfSweepV4PPORunnerCfg(InspireShelfSweepV2PPORunnerCfg):
    experiment_name = "UR5e_shelf_sweep_inspire_right_v4"
