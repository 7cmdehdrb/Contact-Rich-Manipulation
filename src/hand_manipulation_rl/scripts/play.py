#!/usr/bin/env python3
"""Run a local RSL-RL checkpoint on the standalone blind-sweeping task."""

from __future__ import annotations

import argparse
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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--task",
    choices=(BASE_TASK_ID, APPROACH_TASK_ID),
    default=BASE_TASK_ID,
    help="Registered blind-sweep task variant used by the checkpoint.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Stop after this many policy steps; zero runs until the app closes.",
)
parser.add_argument("--real-time", action="store_true")
parser.add_argument(
    "--disable-markers",
    action="store_true",
    help="Hide Target, Goal, and virtual hand-center EEF markers.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
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


def _make_task_configs():
    if args.task == REGISTERED_APPROACH_TASK_ID:
        return BlindSweepApproachEnvCfg(), BlindSweepApproachPPORunnerCfg()
    if args.task == REGISTERED_BASE_TASK_ID:
        return BlindSweepEnvCfg(), BlindSweepReferencePPORunnerCfg()
    raise ValueError(f"Unsupported task: {args.task}")


def main() -> None:
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 0:
        raise ValueError("--steps cannot be negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg, agent_cfg = _make_task_configs()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_vis = not args.disable_markers
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
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        compatible_checkpoint = handle_deprecated_rsl_rl_checkpoint(
            str(checkpoint), installed_version
        )
        print(f"[INFO] Loading model checkpoint: {compatible_checkpoint}")
        runner.load(compatible_checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        observation = wrapped_env.get_observations()
        step_count = 0
        while simulation_app.is_running() and (args.steps == 0 or step_count < args.steps):
            started = time.time()
            with torch.inference_mode():
                action = policy(observation)
                observation, _, dones, _ = wrapped_env.step(action)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                elif hasattr(runner.alg, "policy"):
                    runner.alg.policy.reset(dones)
            step_count += 1
            sleep_time = env.unwrapped.step_dt - (time.time() - started)
            if args.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
        print(f"[INFO] Completed {step_count} policy steps", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
