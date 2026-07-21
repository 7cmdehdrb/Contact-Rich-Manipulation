"""Registration for the standalone UR5e OSC sweep and Home-return task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Sweep-Object-UR5e-OSC-Independent-v0"
DETAILED_TASK_ID = "Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep_independent.env_cfg:"
            "UR5eOscSweepIndependentEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UR5eOscSweepIndependentPPORunnerCfg"
        ),
    },
)

gym.register(
    id=DETAILED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep_independent.env_cfg_detailed:"
            "UR5eOscSweepIndependentDetailedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_detailed:"
            "UR5eOscSweepIndependentDetailedPPORunnerCfg"
        ),
    },
)
