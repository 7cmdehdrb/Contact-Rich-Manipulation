"""Gym registration for the manager-based UR5e OSC sweep task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg:UR5eOscSweepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eOscSweepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg:UR5eOscSweepEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eOscSweepPPORunnerCfg"
        ),
    },
)
