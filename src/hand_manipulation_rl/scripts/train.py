#!/usr/bin/env python3
"""Train the standalone blind-sweeping task with RSL-RL PPO."""

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

BASE_TASK_ID = "Isaac-Blind-Sweep-Inspire-v0"
APPROACH_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v0"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument(
    "--task",
    choices=(BASE_TASK_ID, APPROACH_TASK_ID),
    default=BASE_TASK_ID,
    help="Registered blind-sweep task variant to train.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Optional local RSL-RL checkpoint from which to resume.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from hand_manipulation_rl import APPROACH_TASK_ID as REGISTERED_APPROACH_TASK_ID  # noqa: E402
from hand_manipulation_rl import TASK_ID as REGISTERED_BASE_TASK_ID  # noqa: E402
from hand_manipulation_rl.agents.rsl_rl_ppo_cfg_approach import (  # noqa: E402
    BlindSweepApproachPPORunnerCfg,
)
from hand_manipulation_rl.agents.rsl_rl_ppo_cfg_02 import (  # noqa: E402
    BlindSweepReferencePPORunnerCfg,
)
from hand_manipulation_rl.env_approach_cfg import BlindSweepApproachEnvCfg  # noqa: E402
from hand_manipulation_rl.env_cfg import BlindSweepEnvCfg  # noqa: E402


def _checkpoint_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def _make_task_configs():
    if args.task == REGISTERED_APPROACH_TASK_ID:
        return BlindSweepApproachEnvCfg(), BlindSweepApproachPPORunnerCfg()
    if args.task == REGISTERED_BASE_TASK_ID:
        return BlindSweepEnvCfg(), BlindSweepReferencePPORunnerCfg()
    raise ValueError(f"Unsupported task: {args.task}")


def main() -> None:
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive")

    env_cfg, agent_cfg = _make_task_configs()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.run_name:
        agent_cfg.run_name = args.run_name

    # AppLauncher owns the single device CLI. Keep simulator tensors and the
    # policy/optimizer on that same device, including CPU-only bring-up.
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = (log_root / run_name).resolve()
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args.task, cfg=env_cfg)
    start_time = time.time()
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        if args.checkpoint is not None:
            checkpoint = _checkpoint_path(args.checkpoint)
            print(f"[INFO] Resuming checkpoint: {checkpoint}")
            runner.load(str(checkpoint))

        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        print(f"Training time: {time.time() - start_time:.2f} seconds", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
