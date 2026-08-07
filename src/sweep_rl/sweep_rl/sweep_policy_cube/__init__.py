"""Gym registration for the Sweep-Policy-derived single-cube task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.sweep_policy_cube.env_cfg:SweepPolicyCubeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:SweepPolicyCubePPORunnerCfg"
        ),
    },
)

__all__ = ["TASK_ID"]
