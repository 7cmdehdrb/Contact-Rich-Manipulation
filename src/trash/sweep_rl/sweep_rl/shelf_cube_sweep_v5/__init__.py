"""Gym registration for the standalone shelf Cube Sweep v5 task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v5"
_RSL_CFG = f"{agents.__name__}.rsl_rl_ppo_cfg:ShelfCubeSweepV5PPORunnerCfg"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_sweep_v5.env_cfg:ShelfCubeSweepV5EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG,
        },
    )

if PLAY_TASK_ID not in gym.registry:
    gym.register(
        id=PLAY_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_sweep_v5.env_cfg:"
                "ShelfCubeSweepV5PlayEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG,
        },
    )

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
