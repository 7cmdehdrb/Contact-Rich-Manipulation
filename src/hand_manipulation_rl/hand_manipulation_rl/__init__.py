"""Independent blind-sweeping reinforcement-learning components.

The simulator-facing environment is intentionally kept separate from the pure
math and MDP helpers exposed here.
"""

TASK_ID = "Isaac-Blind-Sweep-Inspire-v0"
APPROACH_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v0"
APPROACH_V1_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v1"


def _register_task() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        # Keep simulator-independent math/sensor modules importable for their
        # pure-torch contract tests. Isaac Lab environments always provide Gym.
        return
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="hand_manipulation_rl.env:BlindSweepEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hand_manipulation_rl.env_cfg:BlindSweepEnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_rl.agents.rsl_rl_ppo_cfg_02:"
                    "BlindSweepReferencePPORunnerCfg"
                ),
            },
        )
    if APPROACH_TASK_ID not in gym.registry:
        gym.register(
            id=APPROACH_TASK_ID,
            entry_point="hand_manipulation_rl.env_approach:BlindSweepApproachEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_rl.env_approach_cfg:BlindSweepApproachEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_rl.agents.rsl_rl_ppo_cfg_approach:"
                    "BlindSweepApproachPPORunnerCfg"
                ),
            },
        )
    if APPROACH_V1_TASK_ID not in gym.registry:
        gym.register(
            id=APPROACH_V1_TASK_ID,
            entry_point="hand_manipulation_rl.env_v1:BlindSweepApproachV1Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "hand_manipulation_rl.env_v1_cfg:BlindSweepApproachV1EnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "hand_manipulation_rl.agents.rsl_rl_ppo_cfg_v1:"
                    "BlindSweepApproachV1PPORunnerCfg"
                ),
            },
        )


_register_task()

__all__ = ["APPROACH_TASK_ID", "APPROACH_V1_TASK_ID", "TASK_ID"]
