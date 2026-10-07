#!/usr/bin/env python3
"""Train the right-only Inspire shelf-sweeping task with reference RSL-RL PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sweep_inspire_rl import TASK_ID, TASK_IDS, TASK_V1_ID
OBJECT_NAMES = ("bottle_1", "cup_1", "cup_2", "mug_1", "mug_2", "can_1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASK_IDS, default=TASK_ID)
parser.add_argument("--object-name", choices=OBJECT_NAMES, default="cup_1")
parser.add_argument("--num_envs", "--num-envs", type=int, default=None)
parser.add_argument("--max_iterations", "--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run-name", default="")
parser.add_argument("--checkpoint", type=Path, default=None, help="Resume from a local RSL-RL checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from sweep_inspire_rl.agents.rsl_rl_ppo_cfg import InspireShelfSweepPPORunnerCfg, InspireShelfSweepV1PPORunnerCfg  # noqa: E402
from sweep_inspire_rl.env_cfg import InspireShelfSweepEnvCfg, InspireShelfSweepV1EnvCfg  # noqa: E402


def main() -> None:
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive")
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg_type = InspireShelfSweepV1EnvCfg if args.task == TASK_V1_ID else InspireShelfSweepEnvCfg
    agent_cfg_type = InspireShelfSweepV1PPORunnerCfg if args.task == TASK_V1_ID else InspireShelfSweepPPORunnerCfg
    env_cfg = env_cfg_type(object_name=args.object_name)
    agent_cfg = agent_cfg_type()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.run_name:
        agent_cfg.run_name = args.run_name
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name / run_name).resolve()
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args.task, cfg=env_cfg)
    started = time.time()
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        runner.add_git_repo_to_log(__file__)
        if checkpoint is not None:
            print(f"[INFO] Resuming checkpoint: {checkpoint}")
            runner.load(str(checkpoint))
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {time.time() - started:.2f} seconds", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
