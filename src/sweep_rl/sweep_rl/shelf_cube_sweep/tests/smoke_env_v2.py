"""Isaac Lab smoke test for the slower-push v2 task."""

from __future__ import annotations

import argparse
import traceback

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
    CUBE_DEPTH,
    CUBE_WIDTH,
    PRE_REACH_Z_OFFSET,
)
from sweep_rl.shelf_cube_sweep.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UR5eGripperShelfCubeSweepV2PPORunnerCfg,
)
from sweep_rl.shelf_cube_sweep.env_cfg import (  # noqa: E402
    ARM_TARGET_MAX_DELTA_V2,
    ARM_VELOCITY_LIMIT_V2,
    CUBE_COM_OFFSET_V1,
    MAX_EPISODE_STEPS_V2,
    PRE_REACH_X_OFFSET,
    PUSH_GOAL_OFFSET,
    PUSH_VELOCITY_PENALTY_MIN_V2,
    PUSH_VELOCITY_REWARD_MIN_V2,
    SHELF_COLLISION_WEIGHT_V2,
    SHELF_FLOOR_SURFACE_HEIGHTS_V2,
    UR5eGripperShelfCubeSweepEnvCfgV2,
    WRIST_TARGET_MAX_DELTA_V2,
    WRIST_VELOCITY_LIMIT_V2,
)


TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2"


def main() -> None:
    cfg = UR5eGripperShelfCubeSweepEnvCfgV2()
    agent_cfg = UR5eGripperShelfCubeSweepV2PPORunnerCfg()
    assert agent_cfg.clip_actions is None
    assert agent_cfg.actor.distribution_cfg.init_std == 1.0
    assert agent_cfg.actor.distribution_cfg.std_type == "scalar"
    assert agent_cfg.algorithm.num_learning_epochs == 8
    assert agent_cfg.algorithm.learning_rate == 1.0e-3
    assert agent_cfg.algorithm.schedule == "adaptive"
    assert agent_cfg.algorithm.entropy_coef == 0.01
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    # The robot and Cube are supported by the Shelf; omit the remote default
    # GroundPlane USD so this smoke test can run without external asset access.
    cfg.scene.ground = None
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 29)
        assert unwrapped.max_episode_length == MAX_EPISODE_STEPS_V2

        cube_spawn = unwrapped.cfg.scene.target_object.spawn
        assert cube_spawn.size == (CUBE_WIDTH, CUBE_DEPTH, 0.20)
        assert cube_spawn.center_of_mass == CUBE_COM_OFFSET_V1

        shelf_rigid_props = unwrapped.cfg.scene.shelf.spawn.rigid_props
        assert shelf_rigid_props.kinematic_enabled is True
        assert shelf_rigid_props.disable_gravity is True

        velocity_limits = unwrapped.cfg.scene.robot.actuators["arm"].velocity_limit_sim
        assert velocity_limits["shoulder_pan_joint"] == ARM_VELOCITY_LIMIT_V2
        assert velocity_limits["wrist_3_joint"] == WRIST_VELOCITY_LIMIT_V2
        assert unwrapped.cfg.scene.robot.spawn.activate_contact_sensors is False
        assert unwrapped.cfg.scene.robot.spawn.articulation_props is None

        action_term = unwrapped.action_manager.get_term("arm_action")
        expected_max_delta = torch.tensor(
            [
                ARM_TARGET_MAX_DELTA_V2,
                ARM_TARGET_MAX_DELTA_V2,
                ARM_TARGET_MAX_DELTA_V2,
                WRIST_TARGET_MAX_DELTA_V2,
                WRIST_TARGET_MAX_DELTA_V2,
                WRIST_TARGET_MAX_DELTA_V2,
            ],
            device=unwrapped.device,
        )
        torch.testing.assert_close(action_term._max_delta[0], expected_max_delta)

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
        assert "cube_vy" not in command_term.metrics
        assert "self_collision_force_max" not in command_term.metrics
        assert "self_collision_termination_rate" not in command_term.metrics
        expected_goal_delta = torch.tensor(
            PUSH_GOAL_OFFSET,
            dtype=cube.data.root_pos_w.dtype,
            device=unwrapped.device,
        )
        torch.testing.assert_close(
            command_term.goal_pos_w - cube.data.root_pos_w,
            expected_goal_delta.expand_as(cube.data.root_pos_w),
        )

        reward_manager = unwrapped.reward_manager
        assert "slow_push_speed" not in reward_manager.active_terms
        assert "end_effector_position_distance" not in reward_manager.active_terms
        push_cfg = reward_manager.get_term_cfg("pushing_target")
        assert push_cfg.params["velocity_reward_min"] == PUSH_VELOCITY_REWARD_MIN_V2
        assert push_cfg.params["velocity_penalty_min"] == PUSH_VELOCITY_PENALTY_MIN_V2
        assert push_cfg.func.__name__ == "pushing_target"
        assert "max_abs_y_velocity" not in push_cfg.params
        shelf_collision_cfg = reward_manager.get_term_cfg("shelf_collision")
        assert shelf_collision_cfg.weight == SHELF_COLLISION_WEIGHT_V2
        assert (
            shelf_collision_cfg.params["surface_heights"]
            == SHELF_FLOOR_SURFACE_HEIGHTS_V2
        )
        assert "self_collision_force" not in reward_manager.active_terms
        assert "self_collision_terminal" not in reward_manager.active_terms

        termination_manager = unwrapped.termination_manager
        assert "shelf_tipped_over" not in termination_manager.active_terms
        assert "self_collision" not in termination_manager.active_terms

        actions = torch.zeros(
            args.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        for _ in range(5):
            observations, rewards, terminated, _, _ = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, 29)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert not torch.any(terminated)
        print("shelf_cube_sweep v2 simulator smoke test: passed", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
