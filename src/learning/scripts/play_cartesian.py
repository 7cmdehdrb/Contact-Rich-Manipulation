"""Roll out the Cartesian-action reaching environment with zero actions."""

from __future__ import annotations

import argparse

from src.learning.scripts._common import add_common_args, make_env, rollout_random


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, "src/learning/configs/reach_cartesian.yaml")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    env, _ = make_env("cartesian", args.config, args.num_envs, args.device, args.seed)
    rollout_random(env, args.steps)


if __name__ == "__main__":
    main()
