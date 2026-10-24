"""Independent UR5e/Axia80/Inspire single-cylinder shelf-sweep task."""

import gymnasium as gym

TASK_ID = "Isaac-Inspire-Sweep-Baseline-v0"
CONTACT_SWEEP_TASK_ID = "Isaac-Inspire-Contact-Sweep-v0"
CONTACT_SWEEP_CLUTTER_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-v0"
CONTACT_SWEEP_CLUTTER_V1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-v1"
CONTACT_SWEEP_CLUTTER_V1_CURRICULUM_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1"
CONTACT_SWEEP_CLUTTER_V2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-v2"
CONTACT_SWEEP_CLUTTER_FT_TEST_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-FT-Test-v0"
CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v0"
CONTACT_SWEEP_CLUTTER_FUSION_V3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v3"
CONTACT_SWEEP_CLUTTER_FUSION_V1_0_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0"
CONTACT_SWEEP_CLUTTER_FUSION_V2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v2"
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
DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v3"
DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2"

OMNI_RENEWAL_OSC_TASK_ID = "Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0"


CONTACT_SWEEP_CLUTTER_FUSION_V1_1_1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1_1"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_1_1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_1_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_1_1PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1_2"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_1_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_1_2PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_4_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_4"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_4_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_4PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_3PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_3_1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3_1"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_3_1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_3_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_3_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_3_1PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_1PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_2"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V1_0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV1_0EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV1_0PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionV3PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs:ContactSweepClutterFusionEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_ppo_cfg:ContactSweepClutterFusionPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FT_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FT_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter:ContactSweepClutterFTTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_envs:ContactSweepClutterFTTestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_ppo_cfg:ContactSweepClutterV1CurriculumPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V1_CURRICULUM_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V1_CURRICULUM_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter:ContactSweepClutterV1CurriculumEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_envs:ContactSweepClutterV1CurriculumEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_ppo_cfg:ContactSweepClutterV1CurriculumPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter:ContactSweepClutterV2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_envs:ContactSweepClutterV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_ppo_cfg:ContactSweepClutterV2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter:ContactSweepClutterV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_envs:ContactSweepClutterV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_ppo_cfg:ContactSweepClutterV1PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter:ContactSweepClutterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_envs:ContactSweepClutterEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_ppo_cfg:ContactSweepClutterPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep:ContactSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_envs:ContactSweepEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_ppo_cfg:ContactSweepPPORunnerCfg",
        },
    )

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_envs:InspireSweepEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if REALISTIC_PHYSICS_TASK_ID not in gym.registry:
    gym.register(
        id=REALISTIC_PHYSICS_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_envs:InspireSweepRealisticPhysicsEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepRealisticPhysicsPPORunnerCfg",
        },
    )

if INITIAL_PRIVILEGED_TASK_ID not in gym.registry:
    gym.register(
        id=INITIAL_PRIVILEGED_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_envs:InspireSweepRealisticInitialPrivilegedEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepRealisticInitialPrivilegedPPORunnerCfg",
        },
    )

if OMNI_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V1_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V2_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if DEX_OMNI_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_OMNI_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireDexOmniEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V3_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_V4_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_V4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_envs:InspireSweepOmniV4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if OMNI_RENEWAL_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=OMNI_RENEWAL_OSC_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireSweepOmniRenewalOSCEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if DEX_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepPPORunnerCfg",
        },
    )

if TACTILE_TASK_ID not in gym.registry:
    gym.register(
        id=TACTILE_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_envs:InspireSweepTactileEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_ppo_cfg:InspireSweepTactilePPORunnerCfg",
        },
    )

if TACTILE_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=TACTILE_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireTactileSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireTactileSweepOmniRenewalPPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexTactileSweepOmniRenewalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalPPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexTactileSweepOmniRenewalV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV1PPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexTactileSweepOmniRenewalV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV2PPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexTactileSweepOmniRenewalV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV3PPORunnerCfg",
        },
    )

if DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_omni_renewal_envs:InspireDexTactileSweepOmniRenewalV2OSCEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV2OSCPPORunnerCfg",
        },
    )

__all__ = [
    "CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID",
    "CONTACT_SWEEP_TASK_ID",
    "CONTACT_SWEEP_CLUTTER_TASK_ID",
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
    "DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID",
    "DEX_TACTILE_OMNI_RENEWAL_V2_OSC_TASK_ID",
]


CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V0_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v0"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs:FusionHoldingV0EnvCfg",
            "rsl_rl_cfg_entry_point":"inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_ppo_cfg:FusionHoldingV0PPORunnerCfg",
        },
    )


CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V0_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v0"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs:FusionHoldingOSCV0EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_ppo_cfg:FusionHoldingOSCV0PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V0_TASK_ID = 'Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v0'
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V0_TASK_ID,
        entry_point='inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point':'inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs:FusionHoldingOSCObjectRandomV0EnvCfg',
            'rsl_rl_cfg_entry_point':'inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_object_random_ppo_cfg:FusionHoldingOSCObjectRandomV0PPORunnerCfg',
        },
    )


CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v1"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs:FusionHoldingV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_ppo_cfg:FusionHoldingV1PPORunnerCfg",
        },
    )


CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v1"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs:FusionHoldingOSCV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_ppo_cfg:FusionHoldingOSCV1PPORunnerCfg",
        },
    )


CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v1"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs:FusionHoldingOSCObjectRandomV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_object_random_ppo_cfg:FusionHoldingOSCObjectRandomV1PPORunnerCfg",
        },
    )

__all__ += [
    "CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V1_TASK_ID",
    "CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V1_TASK_ID",
    "CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V1_TASK_ID",
]

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v2"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs:FusionHoldingOSCV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_ppo_cfg:FusionHoldingOSCV2PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V2_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v2"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs:FusionHoldingV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_ppo_cfg:FusionHoldingV2PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V2_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OBJECT_RANDOM_TEST_V0_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-ObjectRandom-Test-v0"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OBJECT_RANDOM_TEST_V0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OBJECT_RANDOM_TEST_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs:FusionHoldingObjectRandomTestV0EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_object_random_ppo_cfg:FusionHoldingObjectRandomTestV0PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OBJECT_RANDOM_TEST_V0_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v3"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs:FusionHoldingV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_ppo_cfg:FusionHoldingV3PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V3_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v3"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs:FusionHoldingOSCV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_ppo_cfg:FusionHoldingOSCV3PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V3_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v2"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs:FusionHoldingOSCObjectRandomV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_object_random_ppo_cfg:FusionHoldingOSCObjectRandomV2PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V2_TASK_ID")
