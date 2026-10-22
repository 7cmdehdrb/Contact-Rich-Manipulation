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
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1_1.env_cfg:ContactSweepClutterFusionV1_1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1_1.ppo_cfg:ContactSweepClutterFusionV1_1_1PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1_2"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_1_2_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1_2.env_cfg:ContactSweepClutterFusionV1_1_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1_2.ppo_cfg:ContactSweepClutterFusionV1_1_2PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_3_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_3.env_cfg:ContactSweepClutterFusionV1_3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_3.ppo_cfg:ContactSweepClutterFusionV1_3PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_1"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_1_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1.env_cfg:ContactSweepClutterFusionV1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_1.ppo_cfg:ContactSweepClutterFusionV1_1PPORunnerCfg",
        },
    )

CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_2"
if CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_2_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_2.env_cfg:ContactSweepClutterFusionV1_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_2.ppo_cfg:ContactSweepClutterFusionV1_2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V1_0_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V1_0_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.env_cfg:ContactSweepClutterFusionV1_0EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v1_0.ppo_cfg:ContactSweepClutterFusionV1_0PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v2.env_cfg:ContactSweepClutterFusionV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v2.ppo_cfg:ContactSweepClutterFusionV2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_V3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v3.env_cfg:ContactSweepClutterFusionV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion_v3.ppo_cfg:ContactSweepClutterFusionV3PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion.env_cfg:ContactSweepClutterFusionEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_fusion.ppo_cfg:ContactSweepClutterFusionPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_FT_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FT_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_ft_test.env:ContactSweepClutterFTTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_ft_test.env_cfg:ContactSweepClutterFTTestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v1_curriculum.ppo_cfg:ContactSweepClutterV1CurriculumPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V1_CURRICULUM_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V1_CURRICULUM_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_v1_curriculum.env:ContactSweepClutterV1CurriculumEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v1_curriculum.env_cfg:ContactSweepClutterV1CurriculumEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v1_curriculum.ppo_cfg:ContactSweepClutterV1CurriculumPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V2_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_v2.env:ContactSweepClutterV2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v2.env_cfg:ContactSweepClutterV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v2.ppo_cfg:ContactSweepClutterV2PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_V1_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter_v1.env:ContactSweepClutterV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v1.env_cfg:ContactSweepClutterV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter_v1.ppo_cfg:ContactSweepClutterV1PPORunnerCfg",
        },
    )

if CONTACT_SWEEP_CLUTTER_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep_clutter.env:ContactSweepClutterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter.env_cfg:ContactSweepClutterEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep_clutter.ppo_cfg:ContactSweepClutterPPORunnerCfg",
        },
    )

if CONTACT_SWEEP_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_TASK_ID,
        entry_point="inspire_sweep_baseline.contact_sweep.env:ContactSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.contact_sweep.env_cfg:ContactSweepEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.contact_sweep.ppo_cfg:ContactSweepPPORunnerCfg",
        },
    )

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

if DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID not in gym.registry:
    gym.register(
        id=DEX_TACTILE_OMNI_RENEWAL_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.env:InspireSweepEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.dex_tactile_omni_renewal_v3_env_cfg:InspireDexTactileSweepOmniRenewalV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agents.tactile_omni_renewal_ppo_cfg:InspireDexTactileSweepOmniRenewalV3PPORunnerCfg",
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
