"""Gym registration for UR5e gripper reaching inside the shelf."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-v0"
PLAY_TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-Play-v0"

_RSL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eGripperShelfReachPPORunnerCfg"
)

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_reach.env_cfg:UR5eGripperShelfReachEnvCfg"
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
                "sweep_rl.shelf_reach.env_cfg:UR5eGripperShelfReachEnvCfg_PLAY"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG,
        },
    )

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
