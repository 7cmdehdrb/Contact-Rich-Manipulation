#!/usr/bin/env python3
"""Replay a local RSL-RL manipulation checkpoint with task and EEF markers."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
TASK_ID = "Isaac-Hand-Manipulation-Test-v0"
CONTACT_TASK_ID = "Isaac-Hand-Manipulation-Contact-v0"
PUSH_TASK_ID = "Isaac-Hand-Manipulation-Push-v0"
PUSH_V1_TASK_ID = "Isaac-Hand-Manipulation-Push-v1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(TASK_ID, CONTACT_TASK_ID, PUSH_TASK_ID, PUSH_V1_TASK_ID), default=TASK_ID)
parser.add_argument("--num_envs", "--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; zero runs until app exit.")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--disable-markers", action="store_true")
AppLauncher.add_app_launcher_args(parser)
# AppLauncher probes existing arguments before adding its help flag. Register
# required application arguments afterward so plain --help remains available.
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
app_launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import hand_manipulation_test  # noqa: E402, F401 -- registers the Gym tasks


def main() -> None:
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 0:
        raise ValueError("--steps cannot be negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.target_position.debug_vis = not args.disable_markers
    env_cfg.scene.ee_frame.debug_vis = not args.disable_markers
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    env_cfg.log_dir = str(checkpoint.parent)
    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    env = gym.make(args.task, cfg=env_cfg)
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        compatible_checkpoint = handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed_version)
        print(f"[INFO] Loading model checkpoint: {compatible_checkpoint}", flush=True)
        runner.load(compatible_checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        observation = wrapped_env.get_observations()
        count = 0
        while simulation_app.is_running() and (args.steps == 0 or count < args.steps):
            started = time.monotonic()
            with torch.inference_mode():
                action = policy(observation)
                observation, _, dones, _ = wrapped_env.step(action)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                elif hasattr(runner.alg, "policy"):
                    runner.alg.policy.reset(dones)
            count += 1
            wait = env.unwrapped.step_dt - (time.monotonic() - started)
            if args.real_time and wait > 0.0:
                time.sleep(wait)
        print(f"[INFO] Completed {count} policy steps", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
