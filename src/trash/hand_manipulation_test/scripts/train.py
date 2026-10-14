#!/usr/bin/env python3
"""Train a standalone UR5e–Inspire manipulation task with RSL-RL PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import shutil
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
parser.add_argument("--num_envs", "--num-envs", type=int, default=None)
parser.add_argument("--max_iterations", "--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--checkpoint", type=Path, help="Resume a local checkpoint for the selected task.")
markers = parser.add_mutually_exclusive_group()
markers.add_argument("--disable-markers", action="store_true")
markers.add_argument("--enable-markers", action="store_true", help="Keep markers enabled during headless training.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import hand_manipulation_test  # noqa: E402, F401 -- registers the Gym tasks


def main() -> None:
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.run_name:
        agent_cfg.run_name = args.run_name
    # Avoid updating thousands of USD marker instances in headless rollouts.
    if args.disable_markers or (args.headless and not args.enable_markers):
        env_cfg.commands.target_position.debug_vis = False
        env_cfg.scene.ee_frame.debug_vis = False
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name / run_name).resolve()
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    started = time.monotonic()
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        runner.add_git_repo_to_log(__file__)
        if checkpoint is not None:
            print(f"[INFO] Resuming checkpoint: {checkpoint}", flush=True)
            runner.load(str(checkpoint))
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        # git diff omits untracked task sources. Save the small Python/config
        # source tree once so reward and reset implementations are reproducible.
        source_dir = log_dir / "source"
        for folder in ("hand_manipulation_test", "scripts", "tests"):
            for source in (PACKAGE_ROOT / folder).rglob("*.py"):
                relative = source.relative_to(PACKAGE_ROOT)
                if "__pycache__" in relative.parts or "logs" in relative.parts:
                    continue
                destination = source_dir / source.relative_to(PACKAGE_ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
        )
        print(f"[INFO] Training completed in {time.monotonic() - started:.2f} seconds", flush=True)
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
        # Kit's fast shutdown exits the process from close(). Set its public
        # quit status first so an exception cannot become a successful run.
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
