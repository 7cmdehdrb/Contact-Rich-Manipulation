"""Isaac Lab smoke test for the UR5e gripper shelf-reach environment."""

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
from sweep_rl.shelf_reach.env_cfg import (  # noqa: E402
    GRIPPER_JOINT_NAMES,
    TARGET_POS_X,
    TARGET_POS_Y,
    TARGET_POS_Z,
)


TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-v0"


def main() -> None:
    env = gym.make(TASK_ID, num_envs=args.num_envs)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        policy = observations["policy"]
        assert policy.shape == (args.num_envs, 25), policy.shape
        assert unwrapped.action_manager.total_action_dim == 6
        assert set(unwrapped.scene.rigid_objects) == {"shelf"}
        assert unwrapped.scene.rigid_object_collections == {}

        command = unwrapped.command_manager.get_command("ee_pose")
        assert command.shape == (args.num_envs, 7)
        assert torch.all((command[:, 0] >= TARGET_POS_X[0]) & (command[:, 0] <= TARGET_POS_X[1]))
        assert torch.all((command[:, 1] >= TARGET_POS_Y[0]) & (command[:, 1] <= TARGET_POS_Y[1]))
        assert torch.all((command[:, 2] >= TARGET_POS_Z[0]) & (command[:, 2] <= TARGET_POS_Z[1]))
        assert torch.isfinite(command).all()

        tcp_frame = unwrapped.scene["ee_frame"]
        assert tcp_frame.data.target_pos_w.shape == (args.num_envs, 1, 3)
        assert tcp_frame.data.target_quat_w.shape == (args.num_envs, 1, 4)
        assert torch.isfinite(tcp_frame.data.target_pos_w).all()
        assert torch.isfinite(tcp_frame.data.target_quat_w).all()

        robot = unwrapped.scene["robot"]
        gripper_ids, _ = robot.find_joints(
            list(GRIPPER_JOINT_NAMES), preserve_order=True
        )
        assert len(gripper_ids) == len(GRIPPER_JOINT_NAMES)
        torch.testing.assert_close(
            robot.data.joint_pos[:, gripper_ids],
            torch.zeros_like(robot.data.joint_pos[:, gripper_ids]),
            atol=1.0e-3,
            rtol=0.0,
        )

        actions = torch.zeros(
            args.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        for _ in range(5):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 25)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert terminated.shape == (args.num_envs,)
            assert truncated.shape == (args.num_envs,)

        assert set(unwrapped.reward_manager.active_terms) == {
            "end_effector_position_tracking",
            "end_effector_position_tracking_fine_grained",
            "end_effector_orientation_tracking",
            "action_rate",
            "joint_vel",
        }
        assert set(unwrapped.termination_manager.active_terms) == {"time_out"}
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
