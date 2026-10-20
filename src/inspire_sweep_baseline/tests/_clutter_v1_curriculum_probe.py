"""Isaac/PhysX validation of inherited configs and live curriculum application."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401


def values(env):
    manager = env.reward_manager
    return [
        manager.get_term_cfg("pushing_speed").params["free_speed"],
        manager.get_term_cfg("arm_speed_excess").params["free_speed"],
        manager.get_term_cfg("arm_action_rate").weight,
        manager.get_term_cfg("arm_joint_velocity").weight,
    ]


try:
    task = "Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=4)
    parent = parse_env_cfg("Isaac-Inspire-Contact-Sweep-Clutter-v1", device=cfg.sim.device, num_envs=4)
    for name in ("observations", "actions", "commands", "scene", "events", "terminations"):
        assert getattr(cfg, name).to_dict() == getattr(parent, name).to_dict(), name
    original_parent = parent.rewards.to_dict()
    for name in vars(parent.rewards):
        if name not in ("pushing_speed", "arm_speed_excess", "arm_action_rate", "arm_joint_velocity"):
            assert getattr(cfg.rewards, name).to_dict() == getattr(parent.rewards, name).to_dict()
    assert cfg.rewards.arm_action_rate.func is parent.rewards.arm_action_rate.func
    assert cfg.rewards.arm_joint_velocity.func is parent.rewards.arm_joint_velocity.func
    assert cfg.rewards.pushing_speed.params["free_speed"] == .1
    assert cfg.rewards.arm_speed_excess.params["free_speed"] == 1.
    env = gym.make(task, cfg=cfg).unwrapped
    stages = []
    try:
        with torch.inference_mode():
            obs, _ = env.reset(seed=42)
            assert {key: list(value.shape) for key, value in obs.items()} == {
                "policy": [4, 57], "obstacles": [4, 40], "privileged": [4, 3],
            }
            action = torch.zeros(4, 8, device=env.device)
            action[:, -2:] = 1.
            for iteration, expected in [
                (0, [.1, 1., -.03, -.03]),
                (3000, [.09, 1., -.035, -.035]),
                (5000, [.09, .95, -.035, -.035]),
                (15000, [.05, .85, -.055, -.055]),
                (18000, [.05, .85, -.06, -.06]),
                (20000, [.05, .8, -.06, -.06]),
                (90000, [.05, .8, -.06, -.06]),
            ]:
                env.set_learning_iteration(iteration)
                assert values(env) == expected
                # Partial reset cannot advance or undo a stage.
                env.reset(env_ids=torch.tensor([1, 3], device=env.device))
                assert values(env) == expected
                for _ in range(2):
                    obs, reward, _, _, _ = env.step(action)
                    assert torch.isfinite(reward).all()
                    assert all(torch.isfinite(value).all() for value in obs.values())
                assert values(env) == expected
                stages.append({"iteration": iteration, "values": values(env)})
            assert parent.rewards.to_dict() == original_parent
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"task": task, "stages": stages}, indent=2) + "\n")
    finally:
        env.close()
finally:
    launcher.app.close()
