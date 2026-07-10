"""Train or smoke-test the joint-action reaching environment."""

from __future__ import annotations

import argparse

from rsl_rl.runners import OnPolicyRunner

from src.learning.scripts._common import add_common_args, make_env, rollout_random


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "src/learning/configs/reach_joint.yaml")
    parser.add_argument("--iterations", type=int, default=0, help="0 runs only a smoke rollout.")
    parser.add_argument("--smoke-steps", type=int, default=8)
    args = parser.parse_args()

    env, raw_cfg = make_env("joint", args.config, args.num_envs, args.device, args.seed)
    if args.iterations <= 0:
        rollout_random(env, args.smoke_steps)
        return

    train_cfg = raw_cfg["rsl_rl"]
    runner = OnPolicyRunner(env, train_cfg, log_dir="logs/reach_joint", device=train_cfg.get("device", "cpu"))
    runner.learn(args.iterations)


if __name__ == "__main__":
    main()
