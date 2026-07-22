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

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg_wide_randomization:"
            "UR5eOscSweepWideRandomizationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_wide_randomization:"
            "UR5eOscSweepWideRandomizationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg_tactile_localization:"
            "UR5eOscSweepTactileLocalizationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_tactile_localization:"
            "UR5eOscSweepTactileLocalizationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg_constant_velocity:"
            "UR5eOscSweepConstantVelocityEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg_constant_velocity:"
            "UR5eOscSweepConstantVelocityPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg_constant_velocity_upright_random_size:"
            "UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}."
            "rsl_rl_ppo_cfg_constant_velocity_upright_random_size:"
            "UR5eOscSweepConstantVelocityUprightRandomSizePPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-"
        "UprightRandomSize-HomeReturn-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep."
            "env_cfg_constant_velocity_upright_random_size_home:"
            "UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}."
            "rsl_rl_ppo_cfg_constant_velocity_upright_random_size_home:"
            "UR5eOscSweepConstantVelocityUprightRandomSizeHomePPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-"
        "UprightRandomSize-HomeReturn-Can-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep."
            "env_cfg_constant_velocity_upright_random_size_home_can:"
            "UR5eOscSweepConstantVelocityUprightRandomSizeHomeCanEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}."
            "rsl_rl_ppo_cfg_constant_velocity_upright_random_size_home:"
            "UR5eOscSweepConstantVelocityUprightRandomSizeHomePPORunnerCfg"
        ),
    },
)
