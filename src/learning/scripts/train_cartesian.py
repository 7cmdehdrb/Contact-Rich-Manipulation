"""Train or smoke-test the Cartesian-action reaching environment."""

from __future__ import annotations

import argparse
import copy

from rsl_rl.runners import OnPolicyRunner

from src.learning.scripts._common import add_common_args, make_env, rollout_random


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "src/learning/configs/reach_cartesian.yaml")
    parser.add_argument("--iterations", type=int, default=0, help="0 runs only a smoke rollout.")
    parser.add_argument("--smoke-steps", type=int, default=8)
    args = parser.parse_args()

    env, raw_cfg = make_env("cartesian", args.config, args.num_envs, args.device, args.seed, args.backend)
    if args.iterations <= 0:
        rollout_random(env, args.smoke_steps)
        return

    train_cfg = copy.deepcopy(raw_cfg["rsl_rl"])
    if args.rl_device is not None:
        train_cfg["device"] = args.rl_device
    elif args.device is not None:
        train_cfg["device"] = args.device
    runner = OnPolicyRunner(env, train_cfg, log_dir="logs/reach_cartesian", device=train_cfg.get("device", "cpu"))
    runner.learn(args.iterations)


if __name__ == "__main__":
    main()
