"""Sweep-Policy shelf task with the Axia80 / Inspire robot configuration.

String entry points keep Gym registration independent of simulator startup.
"""

TASK_ID = "Isaac-Sweep-Inspire-Right-OSC-v0"


def _register_task() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="sweep_inspire_rl.env:InspireShelfSweepEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "sweep_inspire_rl.env_cfg:InspireShelfSweepEnvCfg",
                "rsl_rl_cfg_entry_point": (
                    "sweep_inspire_rl.agents.rsl_rl_ppo_cfg:InspireShelfSweepPPORunnerCfg"
                ),
            },
        )


_register_task()

__all__ = ["TASK_ID"]
