"""Unchanged baseline PPO settings with separate tactile Renewal log folders."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import InspireSweepPPORunnerCfg


@configclass
class InspireTactileSweepOmniRenewalPPORunnerCfg(InspireSweepPPORunnerCfg):
    experiment_name = "inspire_tactile_sweep_omni_renewal"


@configclass
class InspireDexTactileSweepOmniRenewalPPORunnerCfg(InspireSweepPPORunnerCfg):
    experiment_name = "inspire_dex_tactile_sweep_omni_renewal"


@configclass
class InspireDexTactileSweepOmniRenewalV1PPORunnerCfg(InspireDexTactileSweepOmniRenewalPPORunnerCfg):
    experiment_name = "inspire_dex_tactile_sweep_omni_renewal_v1"


@configclass
class InspireDexTactileSweepOmniRenewalV2PPORunnerCfg(InspireDexTactileSweepOmniRenewalPPORunnerCfg):
    experiment_name = "inspire_dex_tactile_sweep_omni_renewal_v2"


@configclass
class InspireDexTactileSweepOmniRenewalV2OSCPPORunnerCfg(InspireDexTactileSweepOmniRenewalV2PPORunnerCfg):
    experiment_name = "inspire_dex_tactile_sweep_omni_renewal_v2_osc"
