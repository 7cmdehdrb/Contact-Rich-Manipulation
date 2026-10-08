"""Sweep-Policy shelf task with the Axia80 / Inspire robot configuration.

String entry points keep Gym registration independent of simulator startup.
"""

TASK_ID = "Isaac-Sweep-Inspire-Right-OSC-v0"
TASK_V1_ID = "Isaac-Sweep-Inspire-Right-OSC-v1"
TASK_V2_ID = "Isaac-Sweep-Inspire-Right-OSC-v2"
TASK_V3_ID = "Isaac-Sweep-Inspire-Right-OSC-v3"
TASK_V4_ID = "Isaac-Sweep-Inspire-Right-OSC-v4"
TASK_IDS = (TASK_ID, TASK_V1_ID, TASK_V2_ID, TASK_V3_ID, TASK_V4_ID)


def _register_task() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return
    for task_id, env_cfg, agent_cfg in (
        (TASK_ID, "InspireShelfSweepEnvCfg", "InspireShelfSweepPPORunnerCfg"),
        (TASK_V1_ID, "InspireShelfSweepV1EnvCfg", "InspireShelfSweepV1PPORunnerCfg"),
        (TASK_V2_ID, "InspireShelfSweepV2EnvCfg", "InspireShelfSweepV2PPORunnerCfg"),
        (TASK_V3_ID, "InspireShelfSweepV3EnvCfg", "InspireShelfSweepV3PPORunnerCfg"),
        (TASK_V4_ID, "InspireShelfSweepV4EnvCfg", "InspireShelfSweepV4PPORunnerCfg"),
    ):
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="sweep_inspire_rl.env:InspireShelfSweepEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"sweep_inspire_rl.env_cfg:{env_cfg}",
                "rsl_rl_cfg_entry_point": f"sweep_inspire_rl.agents.rsl_rl_ppo_cfg:{agent_cfg}",
            },
        )


_register_task()

__all__ = ["TASK_ID", "TASK_V1_ID", "TASK_V2_ID", "TASK_V3_ID", "TASK_V4_ID", "TASK_IDS"]
