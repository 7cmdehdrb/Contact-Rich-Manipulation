"""Isaac Lab smoke test for Cube PreReach v1 exponential reward."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import sweep_rl  # noqa: F401, E402


TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1"


def main() -> None:
    env = gym.make(TASK_ID, num_envs=args.num_envs)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 29)
        assert unwrapped.action_manager.total_action_dim == 6

        active_rewards = unwrapped.reward_manager.active_terms
        assert "end_effector_position_tracking" in active_rewards
        assert "end_effector_position_tracking_fine_grained" not in active_rewards
        assert "end_effector_orientation_tracking" in active_rewards
        assert "shelf_collision" in active_rewards

        position_cfg = unwrapped.reward_manager.get_term_cfg(
            "end_effector_position_tracking"
        )
        assert position_cfg.weight == 3.0
        assert position_cfg.func.__name__ == "tcp_position_command_reward_exp"
        assert (
            unwrapped.reward_manager.get_term_cfg("shelf_collision").weight
            == -5.0
        )

        actions = torch.zeros(
            args.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        for _ in range(5):
            observations, rewards, _, _, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 29)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
