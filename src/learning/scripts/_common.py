"""Shared command helpers for learning scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.learning.envs import ReachCartesianEnv, ReachEnvConfig, ReachJointEnv
from src.learning.utils.config import load_yaml, repo_root


def add_common_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", type=Path, default=repo_root() / default_config)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)


def make_env(kind: str, config_path: Path, num_envs: int | None, device: str | None, seed: int | None):
    raw_cfg = load_yaml(config_path)
    cfg = ReachEnvConfig.from_mapping(raw_cfg)
    if num_envs is not None:
        cfg.num_envs = num_envs
    if device is not None:
        cfg.device = device
    if seed is not None:
        cfg.seed = seed
    env = ReachJointEnv(cfg) if kind == "joint" else ReachCartesianEnv(cfg)
    return env, raw_cfg


def rollout_random(env, steps: int) -> None:
    obs = env.get_observations()["policy"]
    print(f"obs_shape={tuple(obs.shape)} num_actions={env.num_actions} num_envs={env.num_envs}")
    for step in range(steps):
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, rewards, dones, extras = env.step(actions)
        if step == steps - 1:
            distance = extras["log"]["/reach/mean_distance"].item()
            print(
                f"step={step + 1} reward_mean={rewards.mean().item():.4f} "
                f"done_rate={dones.float().mean().item():.4f} distance={distance:.4f}"
            )
