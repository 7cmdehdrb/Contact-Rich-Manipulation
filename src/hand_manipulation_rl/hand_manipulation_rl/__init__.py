"""Independent blind-sweeping reinforcement-learning components.

The simulator-facing environment is intentionally kept separate from the pure
math and MDP helpers exposed here.
"""

TASK_ID = "Isaac-Blind-Sweep-Inspire-v0"


def _register_task() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        # Keep simulator-independent math/sensor modules importable for their
        # pure-torch contract tests. Isaac Lab environments always provide Gym.
        return
    if TASK_ID in gym.registry:
        return
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


_register_task()

__all__ = ["TASK_ID"]
