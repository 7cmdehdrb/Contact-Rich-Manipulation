"""Gym registration for Cube-relative UR5e shelf pre-reaching."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0"
PLAY_TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0"

_RSL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:"
    "UR5eGripperShelfCubePreReachPPORunnerCfg"
)

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_pre_reach.env_cfg:"
                "UR5eGripperShelfCubePreReachEnvCfg"
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
                "sweep_rl.shelf_cube_pre_reach.env_cfg:"
                "UR5eGripperShelfCubePreReachEnvCfg_PLAY"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG,
        },
    )

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
