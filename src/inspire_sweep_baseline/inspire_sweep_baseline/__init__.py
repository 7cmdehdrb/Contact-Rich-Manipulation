"""Independent UR5e/Axia80/Inspire single-cylinder shelf-sweep task."""

import gymnasium as gym

TASK_ID = "Isaac-Inspire-Sweep-Baseline-v0"
REALISTIC_PHYSICS_TASK_ID = "Isaac-Inspire-Sweep-Realistic-Physics-v0"

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

__all__ = ["TASK_ID", "REALISTIC_PHYSICS_TASK_ID"]
