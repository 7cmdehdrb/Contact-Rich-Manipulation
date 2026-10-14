#!/usr/bin/env python3
"""Play a shelf-force policy with Isaac Lab's RSL-RL player."""

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
STOCK_PLAYER = (
    REPOSITORY_ROOT
    / "IsaacLab"
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "play.py"
)
if not STOCK_PLAYER.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL player not found: {STOCK_PLAYER}")

sys.path.insert(0, str(STOCK_PLAYER.parent))
runpy.run_path(str(STOCK_PLAYER), run_name="__main__")
