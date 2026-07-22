#!/usr/bin/env python3
"""Train the standalone shelf-force task with Isaac Lab's RSL-RL trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import gymnasium as gym


TASK_ID = "Isaac-Sweep-Shelve-Force-v0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _has_task_argument(arguments: list[str]) -> bool:
    return any(arg == "--task" or arg.startswith("--task=") for arg in arguments)


# Register with string entry points so AppLauncher runs before Isaac modules import.
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "sweep_shelve_force.shelf_force.env_cfg:ShelfForceEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "sweep_shelve_force.shelf_force.rsl_rl_ppo_cfg:"
                "ShelfForcePPORunnerCfg"
            ),
        },
    )

if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STOCK_TRAINER = (
    REPOSITORY_ROOT
    / "IsaacLab"
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "train.py"
)
if not STOCK_TRAINER.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL trainer not found: {STOCK_TRAINER}")

sys.path.insert(0, str(STOCK_TRAINER.parent))
runpy.run_path(str(STOCK_TRAINER), run_name="__main__")
