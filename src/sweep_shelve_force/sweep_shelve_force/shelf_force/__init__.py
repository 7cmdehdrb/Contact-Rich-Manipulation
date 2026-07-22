"""Register the single shelf-force sweep environment."""

import gymnasium as gym

from .env_cfg import ShelfForceEnvCfg
from .rsl_rl_ppo_cfg import ShelfForcePPORunnerCfg

TASK_ID = "Isaac-Sweep-Shelve-Force-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": ShelfForceEnvCfg,
            "rsl_rl_cfg_entry_point": ShelfForcePPORunnerCfg,
        },
    )
