"""Independent UR5e/Axia80/Inspire single-cylinder shelf-sweep task."""

import gymnasium as gym

WRENCH_V0_TASK_ID = "Inspire-Wrench-v0"
if WRENCH_V0_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchPPORunnerCfg",
        },
    )

WRENCH_V0_1_TASK_ID = "Inspire-Wrench-v0_1"
if WRENCH_V0_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V0_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV0_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV0_1PPORunnerCfg",
        },
    )

WRENCH_V1_TASK_ID = "Inspire-Wrench-v1"
if WRENCH_V1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV1PPORunnerCfg",
        },
    )

REVISION_V0_TASK_ID = "Isaac-Inspire-Revision-v0"
if REVISION_V0_TASK_ID not in gym.registry:
    gym.register(
        id=REVISION_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.revision:RevisionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.revision_envs:RevisionEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.revision_ppo_cfg:RevisionPPORunnerCfg",
        },
    )

REVISION_V1_TASK_ID = "Isaac-Inspire-Revision-v1"
REVISION_V1_TEST_TASK_ID = "Isaac-Inspire-Revision-Test-v1"
for task_id, env_cfg, runner_cfg in (
    (REVISION_V1_TASK_ID, "RevisionV1EnvCfg", "RevisionV1PPORunnerCfg"),
    (REVISION_V1_TEST_TASK_ID, "RevisionV1TestEnvCfg", "RevisionV1TestPPORunnerCfg"),
):
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="inspire_sweep_baseline.runtime.revision:RevisionEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"inspire_sweep_baseline.envs.revision_envs:{env_cfg}",
                "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.revision_ppo_cfg:{runner_cfg}",
            },
        )

REVISION_V2_TASK_ID = "Isaac-Inspire-Revision-v2"
REVISION_V2_TEST_TASK_ID = "Isaac-Inspire-Revision-Test-v2"
for task_id, env_cfg, runner_cfg in (
    (REVISION_V2_TASK_ID, "RevisionV2EnvCfg", "RevisionV2PPORunnerCfg"),
    (REVISION_V2_TEST_TASK_ID, "RevisionV2TestEnvCfg", "RevisionV2TestPPORunnerCfg"),
):
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="inspire_sweep_baseline.runtime.revision:RevisionEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"inspire_sweep_baseline.envs.revision_envs:{env_cfg}",
                "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.revision_ppo_cfg:{runner_cfg}",
            },
        )

REVISION_V2_TEST_GT_TASK_ID = "Isaac-Inspire-Revision-Test-v2-GT"
if REVISION_V2_TEST_GT_TASK_ID not in gym.registry:
    gym.register(
        id=REVISION_V2_TEST_GT_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.revision:RevisionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.revision_envs:RevisionV2TestGTEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.revision_ppo_cfg:RevisionV2TestGTPPORunnerCfg",
        },
    )

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
    "REVISION_V0_TASK_ID",
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

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V4_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-v4"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V4_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs:FusionHoldingV4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_ppo_cfg:FusionHoldingV4PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_V4_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V4_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v4"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V4_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs:FusionHoldingOSCV4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_ppo_cfg:FusionHoldingOSCV4PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_V4_TASK_ID")

FINAL_V0_TASK_ID = "Isaac-Inspire-Final-v0"
if FINAL_V0_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalPPORunnerCfg",
        },
    )

__all__.append("FINAL_V0_TASK_ID")

FINAL_V1_TASK_ID = "Isaac-Inspire-Final-v1"
FINAL_V2_TASK_ID = "Isaac-Inspire-Final-v2"
for final_version, final_task_id in ((1, FINAL_V1_TASK_ID), (2, FINAL_V2_TASK_ID)):
    if final_task_id not in gym.registry:
        gym.register(
            id=final_task_id,
            entry_point="inspire_sweep_baseline.runtime.final:FinalV1Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"inspire_sweep_baseline.envs.final_envs:FinalV{final_version}EnvCfg",
                "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.final_ppo_cfg:FinalV{final_version}PPORunnerCfg",
            },
        )
__all__.extend(["FINAL_V1_TASK_ID", "FINAL_V2_TASK_ID"])

FINAL_V1_1_TASK_ID = "Isaac-Inspire-Final-v1-1"
if FINAL_V1_1_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_V1_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalV1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalV1_1PPORunnerCfg",
        },
    )
__all__.append("FINAL_V1_1_TASK_ID")

FINAL_V1_TEST_TASK_ID = "Isaac-Inspire-Final-Test-v1"
if FINAL_V1_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_V1_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalV1TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalV1TestPPORunnerCfg",
        },
    )
__all__.append("FINAL_V1_TEST_TASK_ID")

FINAL_GOAL_ACCURACY_V0_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0"
if FINAL_GOAL_ACCURACY_V0_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyPPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_TASK_ID")

FINAL_GOAL_ACCURACY_V1_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v1"
if FINAL_GOAL_ACCURACY_V1_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV1PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V1_TASK_ID")

FINAL_V2_TEST_TASK_ID = "Isaac-Inspire-Final-Test-v2"
if FINAL_V2_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_V2_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalV2TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalV2PPORunnerCfg",
        },
    )
__all__.append("FINAL_V2_TEST_TASK_ID")

CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V3_TASK_ID = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v3"
if CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V3_TASK_ID not in gym.registry:
    gym.register(
        id=CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion:ContactSweepClutterFusionV1_0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs:FusionHoldingOSCObjectRandomV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.contact_sweep_clutter_fusion_holding_osc_object_random_ppo_cfg:FusionHoldingOSCObjectRandomV3PPORunnerCfg",
        },
    )
__all__.append("CONTACT_SWEEP_CLUTTER_FUSION_HOLDING_OSC_OBJECT_RANDOM_V3_TASK_ID")

FINAL_GOAL_ACCURACY_V0_1_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0_1"
if FINAL_GOAL_ACCURACY_V0_1_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV0_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV0_1PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_1_TASK_ID")

FINAL_GOAL_ACCURACY_V0_2_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0_2"
if FINAL_GOAL_ACCURACY_V0_2_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV0_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV0_2PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_2_TASK_ID")

FINAL_GOAL_ACCURACY_V0_3_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0_3"
if FINAL_GOAL_ACCURACY_V0_3_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV0_3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV0_3PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_3_TASK_ID")

FINAL_GOAL_ACCURACY_V0_2_1_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0_2_1"
if FINAL_GOAL_ACCURACY_V0_2_1_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_2_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyV0_2_1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV0_2_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV0_2_1PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_2_1_TASK_ID")

FINAL_GOAL_ACCURACY_V0_2_2_TASK_ID = "Isaac-Inspire-Final-GoalAccuracy-v0_2_2"
if FINAL_GOAL_ACCURACY_V0_2_2_TASK_ID not in gym.registry:
    gym.register(
        id=FINAL_GOAL_ACCURACY_V0_2_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.final:FinalGoalAccuracyV0_2_2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.final_envs:FinalGoalAccuracyV0_2_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.final_ppo_cfg:FinalGoalAccuracyV0_2_2PPORunnerCfg",
        },
    )
__all__.append("FINAL_GOAL_ACCURACY_V0_2_2_TASK_ID")


WRENCH_V2_TASK_ID = "Inspire-Wrench-v2"
if WRENCH_V2_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV2PPORunnerCfg",
        },
    )
__all__.append("WRENCH_V2_TASK_ID")

WRENCH_V0_2_TASK_ID = "Inspire-Wrench-v0_2"
if WRENCH_V0_2_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V0_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV0_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV0_2PPORunnerCfg",
        },
    )


WRENCH_V0_3_TASK_ID = "Inspire-Wrench-v0_3"
if WRENCH_V0_3_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V0_3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV0_3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV0_3PPORunnerCfg",
        },
    )
__all__.append("WRENCH_V0_3_TASK_ID")


WRENCH_V0_4_TASK_ID = "Inspire-Wrench-v0_4"
if WRENCH_V0_4_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_V0_4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench:WrenchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_envs:WrenchV0_4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_ppo_cfg:WrenchV0_4PPORunnerCfg",
        },
    )
__all__.append("WRENCH_V0_4_TASK_ID")


WRENCH_REFACTOR_TASK_ID = "Inspire-Wrench-Refactor-v0"
if WRENCH_REFACTOR_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorPPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V1_TASK_ID = "Inspire-Wrench-Refactor-v1"
if WRENCH_REFACTOR_V1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V2_TASK_ID = "Inspire-Wrench-Refactor-v2"
if WRENCH_REFACTOR_V2_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV2PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V3_TASK_ID = "Inspire-Wrench-Refactor-v3"
if WRENCH_REFACTOR_V3_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV3PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V3_TEST_TASK_ID = "Inspire-Wrench-Refactor-v3-Test"
if WRENCH_REFACTOR_V3_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V3_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV3TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV3TestPPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V0_1_TASK_ID = "Inspire-Wrench-Refactor-v0_1"
if WRENCH_REFACTOR_V0_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V0_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV0_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV0_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V1_1_TASK_ID = "Inspire-Wrench-Refactor-v1_1"
if WRENCH_REFACTOR_V1_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V1_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV1_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV1_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V2_1_TASK_ID = "Inspire-Wrench-Refactor-v2_1"
if WRENCH_REFACTOR_V2_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V2_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV2_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV2_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_V3_1_TASK_ID = "Inspire-Wrench-Refactor-v3_1"
if WRENCH_REFACTOR_V3_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_V3_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorV3_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorV3_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_TASK_ID = "Inspire-Wrench-Refactor-Relative-v0"
if WRENCH_REFACTOR_RELATIVE_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativePPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2"
if WRENCH_REFACTOR_RELATIVE_V2_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_1_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_1"
if WRENCH_REFACTOR_RELATIVE_V2_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_2_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_2"
if WRENCH_REFACTOR_RELATIVE_V2_2_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_2_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_2EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_2PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_3_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_3"
if WRENCH_REFACTOR_RELATIVE_V2_3_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_3PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_4_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_4"
if WRENCH_REFACTOR_RELATIVE_V2_4_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_4PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_5_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_5"
if WRENCH_REFACTOR_RELATIVE_V2_5_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_5_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_5EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_5PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_6_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_6"
if WRENCH_REFACTOR_RELATIVE_V2_6_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_6_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_6EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_6PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V2_7_TASK_ID = "Inspire-Wrench-Refactor-Relative-v2_7"
if WRENCH_REFACTOR_RELATIVE_V2_7_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V2_7_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV2_7EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV2_7PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_IK_V0_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0"
if WRENCH_REFACTOR_RELATIVE_IK_V0_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeIKV0EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeIKV0PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_IK_V0_1_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_1"
if WRENCH_REFACTOR_RELATIVE_IK_V0_1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeIKV0_1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeIKV0_1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_IK_V1_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v1"
if WRENCH_REFACTOR_RELATIVE_IK_V1_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V1_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeIKV1EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeIKV1PPORunnerCfg",
        },
    )


WRENCH_REFACTOR_RELATIVE_V3_TASK_ID = "Inspire-Wrench-Refactor-Relative-v3"
if WRENCH_REFACTOR_RELATIVE_V3_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_V3_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelativeV3EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelativeV3PPORunnerCfg",
        },
    )

# Direction-only evaluation variants reuse the exact parent runtime and PPO.
for _version, _cfg_version in (("v2_1", "V2_1"), ("v3", "V3")):
    for _direction in ("Right", "Pull", "Left", "Push"):
        _task_id = f"Inspire-Wrench-Refactor-Relative-{_version}-Test-{_direction}"
        if _task_id not in gym.registry:
            gym.register(
                id=_task_id,
                entry_point="inspire_sweep_baseline.runtime.wrench_refactor:WrenchRefactorEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"inspire_sweep_baseline.envs.wrench_refactor_envs:WrenchRefactorRelative{_cfg_version}Test{_direction}EnvCfg",
                    "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg:WrenchRefactorRelative{_cfg_version}PPORunnerCfg",
                },
            )

WRENCH_REFACTOR_RELATIVE_IK_V0_2_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2TestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2TestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_1-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2TestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_1TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_1TestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_RESET_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_1-Reset-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_RESET_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_RESET_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_1ResetTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_1ResetTestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_1ResetTestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_1_RESET_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2TestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2TestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_1_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_1-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_1_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_1_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2TestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_1TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_1TestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_1_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_2-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2TestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_2TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_2TestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_FAST_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_2-Fast-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_FAST_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_FAST_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_2_2FastTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_2FastTestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_2FastTestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_2_FAST_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_3_FAST_TEST_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_3-Fast-Test"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_3_FAST_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_3_FAST_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_2_2FastTestEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_3FastTestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_3FastTestPPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_3_FAST_TEST_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_4_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_4"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_4_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_4_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_2_4Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_4EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_4PPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_4_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_5_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_5"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_5_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_5_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_2_4Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_5EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_5PPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_5_TASK_ID")

WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_6_TASK_ID = "Inspire-Wrench-Refactor-Relative-IK-v0_2_2_6"
if WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_6_TASK_ID not in gym.registry:
    gym.register(
        id=WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_6_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test:WrenchRefactorRelativeIKV0_2_2_4Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs:WrenchRefactorRelativeIKV0_2_2_6EnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.wrench_refactor_relative_ik_test_ppo_cfg:WrenchRefactorRelativeIKV0_2_2_6PPORunnerCfg",
        },
    )
__all__.append("WRENCH_REFACTOR_RELATIVE_IK_V0_2_2_6_TASK_ID")

# Independent Simple family: shared runtime, reward and goal-sampling descendants.
for _suffix, _cfg in (
    ("v0", "SweepSimple"),
    ("Prof-v0", "SweepSimpleProf"),
    ("v1", "SweepSimpleV1"),
    ("Prof-v1", "SweepSimpleProfV1"),
    ("Prof-v2", "SweepSimpleProfV2"),
    ("Hard-v0", "SweepSimpleHard"),
    ("Prof-Hard-v0", "SweepSimpleProfHard"),
):
    _task_id = f"Inspire-Sweep-Simple-{_suffix}"
    _runtime = "SweepSimpleFingerEnv" if _suffix == "Prof-v2" else "SweepSimpleEnv"
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point=f"inspire_sweep_baseline.runtime.sweep_simple:{_runtime}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"inspire_sweep_baseline.envs.sweep_simple_envs:{_cfg}EnvCfg",
                "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:{_cfg}PPORunnerCfg",
            },
        )


SWEEP_SIMPLE_FINGER_TASK_ID = "Inspire-Sweep-Simple-Finger-v0"
if SWEEP_SIMPLE_FINGER_TASK_ID not in gym.registry:
    gym.register(
        id=SWEEP_SIMPLE_FINGER_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleFingerEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_simple_envs:SweepSimpleFingerEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:SweepSimpleFingerPPORunnerCfg",
        },
    )
__all__.append("SWEEP_SIMPLE_FINGER_TASK_ID")


SWEEP_SIMPLE_FINGER_WRENCH_TASK_ID = "Inspire-Sweep-Simple-Finger-v0-Wrench"
if SWEEP_SIMPLE_FINGER_WRENCH_TASK_ID not in gym.registry:
    gym.register(
        id=SWEEP_SIMPLE_FINGER_WRENCH_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleFingerEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_simple_envs:SweepSimpleFingerWrenchEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:SweepSimpleFingerWrenchPPORunnerCfg",
        },
    )
__all__.append("SWEEP_SIMPLE_FINGER_WRENCH_TASK_ID")


SWEEP_SIMPLE_FINGER_V0_TEST_TASK_ID = "Inspire-Sweep-Simple-Finger-v0-Test"
if SWEEP_SIMPLE_FINGER_V0_TEST_TASK_ID not in gym.registry:
    gym.register(
        id=SWEEP_SIMPLE_FINGER_V0_TEST_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleFingerEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_simple_envs:SweepSimpleFingerV0TestEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:SweepSimpleFingerV0TestPPORunnerCfg",
        },
    )
__all__.append("SWEEP_SIMPLE_FINGER_V0_TEST_TASK_ID")


for _version in (1, 2):
    _finger_id = f"Inspire-Sweep-Simple-Finger-v{_version}"
    if _finger_id not in gym.registry:
        gym.register(
            id=_finger_id,
            entry_point=f"inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleFingerV{_version}Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"inspire_sweep_baseline.envs.sweep_simple_envs:SweepSimpleFingerV{_version}EnvCfg",
                "rsl_rl_cfg_entry_point": f"inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:SweepSimpleFingerV{_version}PPORunnerCfg",
            },
        )


SWEEP_SIMPLE_SEMI_PROF_TASK_ID = "Inspire-Sweep-Simple-Semi-Prof-v0"
if SWEEP_SIMPLE_SEMI_PROF_TASK_ID not in gym.registry:
    gym.register(
        id=SWEEP_SIMPLE_SEMI_PROF_TASK_ID,
        entry_point="inspire_sweep_baseline.runtime.sweep_simple:SweepSimpleSemiProfEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "inspire_sweep_baseline.envs.sweep_simple_envs:SweepSimpleSemiProfEnvCfg",
            "rsl_rl_cfg_entry_point": "inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:SweepSimpleSemiProfPPORunnerCfg",
        },
    )
__all__.append("SWEEP_SIMPLE_SEMI_PROF_TASK_ID")


# Separate raw-action experiments preserve each original task for playback.
for _raw_suffix, _cfg_prefix, _runtime in (
    ('Finger-Raw-v1', 'SweepSimpleFingerRawV1', 'SweepSimpleFingerV1Env'),
    ('Finger-Raw-v2', 'SweepSimpleFingerRawV2', 'SweepSimpleFingerV2Env'),
    ('Semi-Prof-Raw-v0', 'SweepSimpleSemiProfRaw', 'SweepSimpleSemiProfEnv'),
):
    _raw_id = f'Inspire-Sweep-Simple-{_raw_suffix}'
    if _raw_id not in gym.registry:
        gym.register(
            id=_raw_id,
            entry_point=f'inspire_sweep_baseline.runtime.sweep_simple:{_runtime}',
            disable_env_checker=True,
            kwargs={
                'env_cfg_entry_point': f'inspire_sweep_baseline.envs.sweep_simple_envs:{_cfg_prefix}EnvCfg',
                'rsl_rl_cfg_entry_point': f'inspire_sweep_baseline.agent.sweep_simple_ppo_cfg:{_cfg_prefix}PPORunnerCfg',
            },
        )
