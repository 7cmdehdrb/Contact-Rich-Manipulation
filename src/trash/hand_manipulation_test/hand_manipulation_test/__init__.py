"""Standalone manager-based UR5e–Inspire reaching and palm-contact tasks."""

TASK_ID = "Isaac-Hand-Manipulation-Test-v0"
CONTACT_TASK_ID = "Isaac-Hand-Manipulation-Contact-v0"
PUSH_TASK_ID = "Isaac-Hand-Manipulation-Push-v0"
PUSH_V1_TASK_ID = "Isaac-Hand-Manipulation-Push-v1"


def _register_task() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        # Pure-torch math and sensor helpers also work without a simulator.
        return
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="hand_manipulation_test.env:HandManipulationTestEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_test.config.ur5e.reach_env_cfg:UR5eInspireReachEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_test.agents.rsl_rl_ppo_cfg:HandManipulationTestPPORunnerCfg"
                ),
            },
        )
    if CONTACT_TASK_ID not in gym.registry:
        gym.register(
            id=CONTACT_TASK_ID,
            entry_point="hand_manipulation_test.contact_env:HandManipulationContactEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_test.config.ur5e.contact_env_cfg:UR5eInspireContactEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_test.agents.rsl_rl_contact_ppo_cfg:ContactPPORunnerCfg"
                ),
            },
        )
    if PUSH_TASK_ID not in gym.registry:
        gym.register(
            id=PUSH_TASK_ID,
            entry_point="hand_manipulation_test.push_env:HandManipulationPushEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_test.config.ur5e.push_env_cfg:UR5eInspirePushEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_test.agents.rsl_rl_push_ppo_cfg:PushPPORunnerCfg"
                ),
            },
        )
    if PUSH_V1_TASK_ID not in gym.registry:
        gym.register(
            id=PUSH_V1_TASK_ID,
            entry_point="hand_manipulation_test.push_v1_env:HandManipulationPushV1Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_test.config.ur5e.push_v1_env_cfg:UR5eInspirePushV1EnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_test.agents.rsl_rl_push_v1_ppo_cfg:PushV1PPORunnerCfg"
                ),
            },
        )


_register_task()
