"""Inherited PPO with explicit asymmetric Actor/Critic observation routes."""

from isaaclab.utils import configclass
from .realistic_physics_ppo_cfg import InspireSweepRealisticPhysicsPPORunnerCfg


@configclass
class InspireSweepRealisticInitialPrivilegedPPORunnerCfg(InspireSweepRealisticPhysicsPPORunnerCfg):
    experiment_name = "inspire_sweep_realistic_initial_privileged"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
