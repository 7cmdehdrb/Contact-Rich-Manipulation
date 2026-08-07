"""Gym registration for the basic single-object sweep task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Sweep-Object-UR5e-Basic-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sweep_rl.sweep_basic.env_cfg:SweepBasicEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SweepBasicPPORunnerCfg"
        ),
    },
)

__all__ = ["TASK_ID"]
