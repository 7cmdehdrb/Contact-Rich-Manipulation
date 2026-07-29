#!/usr/bin/env python3
"""Play the HomeReturn checkpoint in the isolated Can_6 environment."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

TASK_ID = (
    "Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-"
    "UprightRandomSize-HomeReturn-Can-v0"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = str(
    REPOSITORY_ROOT
    / "logs"
    / "rsl_rl"
    / "ur5e_osc_sweep_constant_velocity_upright_random_size_home"
    / "2026-07-19_19-55-35"
    / "model_11999.pt"
)
DEFAULT_TARGET_Z_OFFSET = 0.5 * 0.11913070456885584


def _has_argument(arguments: list[str], name: str) -> bool:
    return any(
        argument == name or argument.startswith(f"{name}=")
        for argument in arguments
    )


custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument("--object_mass", type=float, default=0.35)
custom_parser.add_argument(
    "--target_z_offset",
    type=float,
    default=DEFAULT_TARGET_Z_OFFSET,
)
custom_args, stock_arguments = custom_parser.parse_known_args(sys.argv[1:])

if custom_args.object_mass <= 0.0:
    raise ValueError("--object_mass must be positive.")
if custom_args.target_z_offset < 0.0:
    raise ValueError("--target_z_offset must be non-negative.")

if not _has_argument(stock_arguments, "--task"):
    stock_arguments.extend(("--task", TASK_ID))
if not _has_argument(stock_arguments, "--checkpoint"):
    stock_arguments.extend(("--checkpoint", DEFAULT_CHECKPOINT))
if not _has_argument(stock_arguments, "--num_envs"):
    stock_arguments.extend(("--num_envs", "1"))

stock_arguments.extend(
    (
        f"env.events.set_target_mass.params.mass={custom_args.object_mass:.17g}",
        "env.observations.policy.initial_target_pose.params.z_offset="
        f"{custom_args.target_z_offset:.17g}",
        "env.observations.policy.current_target_pose.params.z_offset="
        f"{custom_args.target_z_offset:.17g}",
    )
)
sys.argv = [sys.argv[0], *stock_arguments]

print(
    "[INFO] Can playback override: "
    f"mass={custom_args.object_mass:.6f} kg, "
    f"target observation z_offset={custom_args.target_z_offset:.6f} m",
    flush=True,
)

stock_player = (
    REPOSITORY_ROOT
    / "IsaacLab"
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "play.py"
)
if not stock_player.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL player not found: {stock_player}")

sys.path.insert(0, str(stock_player.parent))
runpy.run_path(str(stock_player), run_name="__main__")
