"""Independent UR5e/Axia80/Inspire single-cylinder shelf-sweep task."""

import gymnasium as gym

TASK_ID = "Isaac-Inspire-Sweep-Baseline-v0"
REALISTIC_PHYSICS_TASK_ID = "Isaac-Inspire-Sweep-Realistic-Physics-v0"
INITIAL_PRIVILEGED_TASK_ID = (
    "Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0"
)
OMNI_TASK_ID = "Isaac-Inspire-Sweep-Omni-v0"
OMNI_V1_TASK_ID = "Isaac-Inspire-Sweep-Omni-v1"
OMNI_V2_TASK_ID = "Isaac-Inspire-Sweep-Omni-v2"
OMNI_V3_TASK_ID = "Isaac-Inspire-Sweep-Omni-v3"
OMNI_V4_TASK_ID = "Isaac-Inspire-Sweep-Omni-v4"

TACTILE_TASK_ID = "Isaac-Inspire-Sweep-Baseline-Tactile-v0"

DEX_OMNI_TASK_ID = "Isaac-Inspire-Dex-Omni-v0"

# ============== Renewal Envs ==============

OMNI_RENEWAL_TASK_ID = "Isaac-Inspire-Sweep-Omni-Renewal-v0"
DEX_OMNI_RENEWAL_TASK_ID = "Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0"

TACTILE_OMNI_RENEWAL_TASK_ID = "Isaac-Inspire-Tactile-Sweep-Omni-Renewal-v0"
DEX_TACTILE_OMNI_RENEWAL_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0"
DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v1"
DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2"
DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2"

OMNI_RENEWAL_OSC_TASK_ID = "Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0"


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.env_cfg:InspireSweepEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if REALISTIC_PHYSICS_TASK_ID not in gym.registry:
    gym.register(
        id=REALISTIC_PHYSICS_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.realistic_physics_env_cfg:InspireSweepRealisticPhysicsEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.realistic_physics_ppo_cfg:InspireSweepRealisticPhysicsPPORunnerCfg",
        },
    )

if INITIAL_PRIVILEGED_TASK_ID not in gym.registry:
    gym.register(
        id=INITIAL_PRIVILEGED_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.initial_privileged_env_cfg:InspireSweepRealisticInitialPrivilegedEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.initial_privileged_ppo_cfg:InspireSweepRealisticInitialPrivilegedPPORunnerCfg",
        },
    )

if OMNI_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V1_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V2_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if DEX_OMNI_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_OMNI_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_omni_env_cfg:InspireDexOmniEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V3_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V4_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V4_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_env_cfg:InspireSweepOmniV4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_renewal_env_cfg:InspireSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_RENEWAL_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_RENEWAL_OSC_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.omni_renewal_osc_env_cfg:InspireSweepOmniRenewalOSCEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if DEX_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_omni_renewal_env_cfg:InspireDexSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.rsl_rl_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if TACTILE_TASK_ID not in gym.registry:
    gym.register(
        id=TACTILE_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.tactile_env_cfg:InspireSweepTactileEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_ppo_cfg:InspireSweepTactilePPORunnerCfg",
        },
    )

if TACTILE_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=TACTILE_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.tactile_omni_renewal_env_cfg:InspireTactileSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireTactileSweepOmniRenewalPPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.tactile_omni_renewal_env_cfg:InspireDexTactileSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalPPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_tactile_omni_renewal_v1_env_cfg:InspireDexTactileSweepOmniRenewalV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV1PPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_tactile_omni_renewal_v2_env_cfg:InspireDexTactileSweepOmniRenewalV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV2PPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_tactile_omni_renewal_v2_osc_env_cfg:InspireDexTactileSweepOmniRenewalV2OSCEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV2OSCPPORunnerCfg",
        },
    )

__all__ = [
    "TASK_ID",
    "REALISTIC_PHYSICS_TASK_ID",
    "INITIAL_PRIVILEGED_TASK_ID",
    "OMNI_TASK_ID",
    "OMNI_V1_TASK_ID",
    "OMNI_V2_TASK_ID",
    "OMNI_V3_TASK_ID",
    "OMNI_V4_TASK_ID",
    "OMNI_RENEWAL_TASK_ID",
    "OMNI_RENEWAL_OSC_TASK_ID",
    "DEX_OMNI_RENEWAL_TASK_ID",
    "DEX_OMNI_TASK_ID",
    "TACTILE_TASK_ID",
    "TACTILE_OMNI_RENEWAL_TASK_ID",
    "DEX_TACTILE_OMNI_RENEWAL_TASK_ID",
    "DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID",
    "DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID",
    "DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID",
]
