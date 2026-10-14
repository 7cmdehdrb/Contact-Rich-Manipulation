#!/usr/bin/env python3
"""Headless Isaac Sim smoke test for the single-stage basic sweep environment."""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import sweep_rl  # noqa: F401, E402
from sweep_rl.sweep_basic import TASK_ID  # noqa: E402
from sweep_rl.sweep_basic import mdp  # noqa: E402
from sweep_rl.sweep_basic.env_cfg import (  # noqa: E402
    EXTERNAL_FINGER_OFFSET,
    SHELF_SURFACE_HEIGHT_OFFSET,
    PRECONTACT_STANDOFF,
    PUSH_STANDOFF,
    SWEEP_DISTANCE,
    TARGET_HEIGHT,
    TARGET_MASS,
    TARGET_MAXIMUM_HEIGHT_CHANGE,
    TARGET_SPEED,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
    SweepBasicEnvCfg,
)


def main() -> None:
    cfg = SweepBasicEnvCfg()
    cfg.scene.num_envs = 8
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        unwrapped = env.unwrapped
        policy = observations["policy"]
        critic = observations["critic"]
        assert policy.shape == (8, 31), policy.shape
        assert critic.shape == (8, 34), critic.shape
        assert torch.isfinite(policy).all()
        assert torch.isfinite(critic).all()
        assert torch.allclose(critic[:, 31:34], torch.zeros_like(critic[:, 31:34]))
        assert unwrapped.action_manager.total_action_dim == 6

        target = unwrapped.scene["target_object"]
        target_position = target.data.root_pos_w - unwrapped.scene.env_origins
        expected_z = SHELF_SURFACE_HEIGHT_OFFSET + 0.5 * TARGET_HEIGHT
        assert unwrapped.cfg.scene.target_object.spawn.size == (
            0.06,
            0.06,
            TARGET_HEIGHT,
        )
        assert unwrapped.cfg.scene.target_object.spawn.mass_props.mass == TARGET_MASS
        assert (
            unwrapped.cfg.terminations.object_drop.params["maximum_height_change"]
            == TARGET_MAXIMUM_HEIGHT_CHANGE
        )
        assert torch.all(
            (target_position[:, 0] >= TARGET_X_RANGE[0])
            & (target_position[:, 0] <= TARGET_X_RANGE[1])
        )
        assert torch.all(
            (target_position[:, 1] >= TARGET_Y_RANGE[0])
            & (target_position[:, 1] <= TARGET_Y_RANGE[1])
        )
        assert torch.allclose(
            target_position[:, 2],
            torch.full_like(target_position[:, 2], expected_z),
            atol=2.0e-3,
        )

        command = unwrapped.command_manager.get_term("target_goal")
        frames = unwrapped.scene["ee_frame"].data.target_pos_w
        assert frames.shape[1] == 4
        assert command.cfg.contact_frame_index == 0
        assert command.cfg.wrist_frame_index == 3
        assert torch.allclose(command.contact_point_pos_w(), frames[:, 0])
        assert torch.allclose(command.wrist_pos_w(), frames[:, 3])
        assert torch.allclose(policy[:, 21:24], command.goal_pos_b)
        assert torch.allclose(critic[:, 21:24], command.goal_pos_b)
        precontact_delta = target.data.root_pos_w - command.precontact_point_w()
        push_delta = target.data.root_pos_w - command.push_point_w()
        expected_precontact_delta = torch.zeros_like(precontact_delta)
        expected_precontact_delta[:, 1] = PRECONTACT_STANDOFF
        expected_precontact_delta[:, 2] = -EXTERNAL_FINGER_OFFSET
        expected_push_delta = torch.zeros_like(push_delta)
        expected_push_delta[:, 1] = PUSH_STANDOFF
        expected_push_delta[:, 2] = -EXTERNAL_FINGER_OFFSET
        assert torch.allclose(
            precontact_delta, expected_precontact_delta, atol=1.0e-6
        )
        assert torch.allclose(push_delta, expected_push_delta, atol=1.0e-6)
        goal_delta = command.goal_pos_w - command.initial_pos_w
        expected_delta = torch.zeros_like(goal_delta)
        expected_delta[:, 1] = SWEEP_DISTANCE
        assert torch.allclose(goal_delta, expected_delta, atol=1.0e-6)
        assert command.cfg.target_speed == TARGET_SPEED
        goal_position = command.goal_pos_w - unwrapped.scene.env_origins
        assert torch.all(
            (goal_position[:, 1] >= 0.28) & (goal_position[:, 1] <= 0.32)
        )

        shelf_cfg = unwrapped.cfg.scene.shelf.spawn.rigid_props
        assert shelf_cfg.kinematic_enabled is True
        assert shelf_cfg.disable_gravity is True
        shelf_contact = unwrapped.scene["shelf_contact"].data.force_matrix_w
        assert shelf_contact is not None
        assert shelf_contact.shape[0] == 8
        assert shelf_contact.shape[-1] == 3
        assert torch.isfinite(shelf_contact).all()
        target_contact = unwrapped.scene[
            "target_gripper_contact"
        ].data.force_matrix_w
        assert target_contact is not None
        assert target_contact.shape[0] == 8
        assert target_contact.shape[-1] == 3
        assert torch.isfinite(target_contact).all()

        actions = torch.zeros((8, 6), device=unwrapped.device)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        assert next_observations["policy"].shape == (8, 31)
        assert next_observations["critic"].shape == (8, 34)
        assert torch.isfinite(next_observations["policy"]).all()
        assert torch.isfinite(next_observations["critic"]).all()
        assert torch.isfinite(rewards).all()
        assert terminated.shape == (8,)
        assert truncated.shape == (8,)

        action_term = unwrapped.action_manager.get_term("arm_action")
        assert torch.all(action_term.raw_actions == 0.0)
        assert torch.all(action_term._gripper_open_targets == 0.0)
        robot = unwrapped.scene["robot"]
        gripper_targets = robot.data.joint_pos_target[:, action_term._gripper_joint_ids]
        assert torch.allclose(gripper_targets, torch.zeros_like(gripper_targets))

        # A policy command is converted once to a current-relative, latched target.
        current_arm_pos = robot.data.joint_pos[:, action_term._joint_ids].clone()
        probe_action = torch.ones((8, 6), device=unwrapped.device)
        action_term.process_actions(probe_action)
        expected_target = current_arm_pos + 0.05
        limits = robot.data.soft_joint_pos_limits[:, action_term._joint_ids]
        expected_target = torch.clamp(
            expected_target, min=limits[..., 0], max=limits[..., 1]
        )
        assert torch.allclose(action_term.processed_actions, expected_target)
        action_term.process_actions(torch.zeros_like(probe_action))

        # Exercise the single-stage success condition at the stopped endpoint.
        target.write_root_pose_to_sim(
            torch.cat((command.goal_pos_w, target.data.root_quat_w), dim=-1)
        )
        target.write_root_velocity_to_sim(torch.zeros_like(target.data.root_vel_w))
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()
        success = mdp.target_reached_and_stopped(
            unwrapped,
            command_name="target_goal",
            endpoint_threshold=0.03,
            speed_threshold=0.02,
        )
        assert torch.all(success)

        print("Sweep basic single-stage smoke test passed.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
