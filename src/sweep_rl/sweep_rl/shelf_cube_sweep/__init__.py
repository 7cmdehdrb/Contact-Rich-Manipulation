"""Gym registration for the inherited UR5e shelf Cube sweep task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0"
TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1"
PLAY_TASK_ID_V1 = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v1"

_RSL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:"
    "UR5eGripperShelfCubeSweepPPORunnerCfg"
)
_RSL_CFG_V1 = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:"
    "UR5eGripperShelfCubeSweepV1PPORunnerCfg"
)

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_sweep.env_cfg:"
                "UR5eGripperShelfCubeSweepEnvCfg"
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
                "sweep_rl.shelf_cube_sweep.env_cfg:"
                "UR5eGripperShelfCubeSweepEnvCfg_PLAY"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG,
        },
    )

if TASK_ID_V1 not in gym.registry:
    gym.register(
        id=TASK_ID_V1,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_sweep.env_cfg:"
                "UR5eGripperShelfCubeSweepEnvCfgV1"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG_V1,
        },
    )

if PLAY_TASK_ID_V1 not in gym.registry:
    gym.register(
        id=PLAY_TASK_ID_V1,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_rl.shelf_cube_sweep.env_cfg:"
                "UR5eGripperShelfCubeSweepEnvCfgV1_PLAY"
            ),
            "rsl_rl_cfg_entry_point": _RSL_CFG_V1,
        },
    )

__all__ = [
    "PLAY_TASK_ID",
    "PLAY_TASK_ID_V1",
    "TASK_ID",
    "TASK_ID_V1",
]
