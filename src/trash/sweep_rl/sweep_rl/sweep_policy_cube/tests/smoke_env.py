"""Isaac Lab smoke test for the Sweep-Policy Cube environment."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import sweep_rl  # noqa: F401, E402


TASK_ID = "Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0"


def main() -> None:
    env = gym.make(TASK_ID, num_envs=args.num_envs)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        policy = observations["policy"]
        assert policy.shape == (args.num_envs, 32), policy.shape
        assert unwrapped.action_manager.total_action_dim == 6
        assert set(unwrapped.scene.rigid_objects) == {"shelf", "target_object"}

        command = unwrapped.command_manager.get_term("target_goal_pos")
        target = unwrapped.scene["target_object"]
        delta = command.goal_pos_w - target.data.root_pos_w
        expected = torch.tensor([0.0, 0.18, 0.0], device=delta.device)
        torch.testing.assert_close(delta, expected.expand_as(delta))

        width_term = unwrapped.observation_manager.get_term_cfg(
            "policy", "target_obj_width"
        )
        assert width_term.params["width"] == 0.06

        actions = torch.zeros(
            args.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        for _ in range(5):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 32)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)

        reward_names = set(unwrapped.reward_manager.active_terms)
        assert "object_collision" not in reward_names
        assert reward_names == {
            "action_rate",
            "joint_vel",
            "end_effector_position_tracking",
            "end_effector_position_tracking_fine_grained",
            "orientation",
        }
        assert set(unwrapped.termination_manager.active_terms) == {"time_out"}
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
