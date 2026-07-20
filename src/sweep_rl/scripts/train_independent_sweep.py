#!/usr/bin/env python3
"""Train the independent OSC sweep and Home-return environment."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TASK_ID = "Isaac-Sweep-Object-UR5e-OSC-Independent-v0"


def _has_task_argument(arguments: list[str]) -> bool:
    return any(
        argument == "--task" or argument.startswith("--task=") for argument in arguments
    )


if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

repository_root = Path(__file__).resolve().parents[3]
stock_trainer = (
    repository_root
    / "IsaacLab"
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "train.py"
)
if not stock_trainer.is_file():
    raise FileNotFoundError(f"IsaacLab RSL-RL trainer not found: {stock_trainer}")

sys.path.insert(0, str(stock_trainer.parent))
runpy.run_path(str(stock_trainer), run_name="__main__")
