"""Isaac Lab smoke test for Cube-relative shelf pre-reaching."""

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
    CUBE_HEIGHT,
    CUBE_MASS,
    CUBE_WIDTH,
    PRE_REACH_Z_OFFSET,
    PUSH_GOAL_OFFSET,
    ROBOT_CONTACT_BODY_PATHS,
)


TASK_ID = "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0"


def main() -> None:
    env = gym.make(TASK_ID, num_envs=args.num_envs)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 29)
        assert unwrapped.action_manager.total_action_dim == 6
        assert set(unwrapped.scene.rigid_objects) == {"shelf", "target_object"}
        assert "shelf_floor_contact" in unwrapped.scene.sensors
        assert "wrist_frame" in unwrapped.scene.sensors

        shelf_floor_contact = unwrapped.scene["shelf_floor_contact"]
        assert shelf_floor_contact.data.force_matrix_w is not None
        assert shelf_floor_contact.data.contact_pos_w is not None
        assert shelf_floor_contact.data.force_matrix_w.shape == (
            args.num_envs,
            1,
            len(ROBOT_CONTACT_BODY_PATHS),
            3,
        )
        assert shelf_floor_contact.data.contact_pos_w.shape == (
            args.num_envs,
            1,
            len(ROBOT_CONTACT_BODY_PATHS),
            3,
        )

        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["target_object"]
        command_b = unwrapped.command_manager.get_command("ee_pose")
        desired_pos_w, _ = math_utils.combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command_b[:, :3],
            command_b[:, 3:7],
        )
        delta = desired_pos_w - cube.data.root_pos_w
        expected = torch.tensor(
            [0.0, -CUBE_WIDTH * BEHIND_WIDTH_SCALE, PRE_REACH_Z_OFFSET],
            dtype=delta.dtype,
            device=delta.device,
        )
        torch.testing.assert_close(delta, expected.expand_as(delta))

        command_term = unwrapped.command_manager.get_term("ee_pose")
        goal_delta = command_term.goal_pos_w - cube.data.root_pos_w
        expected_goal_delta = torch.tensor(
            PUSH_GOAL_OFFSET, dtype=goal_delta.dtype, device=goal_delta.device
        )
        torch.testing.assert_close(
            goal_delta, expected_goal_delta.expand_as(goal_delta)
        )

        width_cfg = unwrapped.observation_manager.get_term_cfg(
            "policy", "cube_width"
        )
        assert width_cfg.params["width"] == CUBE_WIDTH
        assert CUBE_HEIGHT == 0.15
        assert CUBE_MASS == 0.50
        assert "shelf_collision" in unwrapped.reward_manager.active_terms
        shelf_collision_cfg = unwrapped.reward_manager.get_term_cfg(
            "shelf_collision"
        )
        assert shelf_collision_cfg.weight == -0.02
        assert shelf_collision_cfg.params["sensor_name"] == "shelf_floor_contact"
        assert shelf_collision_cfg.params["force_threshold"] == 1.0
        assert shelf_collision_cfg.params["surface_height"] == 1.05
        assert "pushing_target" in unwrapped.reward_manager.active_terms
        pushing_cfg = unwrapped.reward_manager.get_term_cfg("pushing_target")
        assert pushing_cfg.weight == 6.0
        assert pushing_cfg.params["reach_position_threshold"] == 0.10
        assert pushing_cfg.params["reach_orientation_threshold"] == 0.50

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
