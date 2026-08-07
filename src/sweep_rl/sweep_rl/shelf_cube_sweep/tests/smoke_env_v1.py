"""Isaac Lab smoke test for the CubePreReach-v1-based Sweep task."""

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

import isaaclab.utils.math as math_utils
import sweep_rl  # noqa: F401, E402
from sweep_rl.shelf_cube_pre_reach.env_cfg import (  # noqa: E402
    BEHIND_WIDTH_SCALE,
    CUBE_WIDTH,
    PRE_REACH_Z_OFFSET,
)
from sweep_rl.shelf_cube_sweep.env_cfg import (  # noqa: E402
    PRE_REACH_X_OFFSET,
    PUSH_GOAL_OFFSET,
)


TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1"


def main() -> None:
    env = gym.make(TASK_ID, num_envs=args.num_envs)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 29)
        assert "shelf_floor_contact" in unwrapped.scene.sensors
        assert "wrist_frame" in unwrapped.scene.sensors

        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["target_object"]
        command_b = unwrapped.command_manager.get_command("ee_pose")
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command_b[:, :3],
            command_b[:, 3:7],
        )
        reach_delta = desired_pos_w - cube.data.root_pos_w
        expected_reach_delta = torch.tensor(
            [
                PRE_REACH_X_OFFSET,
                -CUBE_WIDTH * BEHIND_WIDTH_SCALE,
                PRE_REACH_Z_OFFSET,
            ],
            dtype=reach_delta.dtype,
            device=reach_delta.device,
        )
        torch.testing.assert_close(
            reach_delta, expected_reach_delta.expand_as(reach_delta)
        )

        command_term = unwrapped.command_manager.get_term("ee_pose")
        goal_delta = command_term.goal_pos_w - cube.data.root_pos_w
        expected_goal_delta = torch.tensor(
            PUSH_GOAL_OFFSET, dtype=goal_delta.dtype, device=goal_delta.device
        )
        torch.testing.assert_close(
            goal_delta, expected_goal_delta.expand_as(goal_delta)
        )

        reward_manager = unwrapped.reward_manager
        assert "end_effector_position_tracking" in reward_manager.active_terms
        assert (
            "end_effector_position_tracking_fine_grained"
            not in reward_manager.active_terms
        )
        assert "end_effector_orientation_tracking" in reward_manager.active_terms
        assert "shelf_collision" in reward_manager.active_terms
        assert "pushing_target" in reward_manager.active_terms
        assert reward_manager.get_term_cfg("end_effector_position_tracking").weight == 3.0
        assert reward_manager.get_term_cfg("end_effector_orientation_tracking").weight == -0.7
        assert reward_manager.get_term_cfg("action_rate").weight == -0.03
        assert reward_manager.get_term_cfg("joint_vel").weight == -0.03
        assert reward_manager.get_term_cfg("pushing_target").weight == 12.0

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
