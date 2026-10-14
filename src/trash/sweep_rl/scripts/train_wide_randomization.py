#!/usr/bin/env python3
"""Train the wide-randomization sweep task without modifying IsaacLab train.py.

The task is registered before delegating to IsaacLab's stock RSL-RL trainer,
so all standard trainer arguments remain available.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import gymnasium as gym


TASK_ID = "Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sweep_rl.osc_sweep.env_cfg_wide_randomization:"
            "UR5eOscSweepWideRandomizationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "sweep_rl.osc_sweep.agents.rsl_rl_ppo_cfg_wide_randomization:"
            "UR5eOscSweepWideRandomizationPPORunnerCfg"
        ),
    },
)


def _has_task_argument(arguments: list[str]) -> bool:
    return any(argument == "--task" or argument.startswith("--task=") for argument in arguments)


if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

repository_root = Path(__file__).resolve().parents[3]
stock_trainer = repository_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not stock_trainer.is_file():
    raise FileNotFoundError(f"IsaacLab RSL-RL trainer not found: {stock_trainer}")

# The stock trainer imports its sibling ``cli_args.py`` as a local module.
sys.path.insert(0, str(stock_trainer.parent))
runpy.run_path(str(stock_trainer), run_name="__main__")
