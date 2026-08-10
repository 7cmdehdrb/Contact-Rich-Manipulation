"""Isaac Lab smoke test for the standalone shelf Cube Sweep v5 task."""

from __future__ import annotations

import argparse
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import sweep_rl  # noqa: F401, E402
from rsl_rl.modules import GaussianDistribution  # noqa: E402
from sweep_rl.shelf_cube_sweep_v5.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    ShelfCubeSweepV5PPORunnerCfg,
)
from sweep_rl.shelf_cube_sweep_v5.env_cfg import (  # noqa: E402
    ARM_MAX_JOINT_SPEED,
    CONTACT_POINT_BEHIND_WIDTH_SCALE,
    CONTACT_POINT_X_OFFSET,
    CONTACT_POINT_Z_OFFSET,
    CUBE_COM_OFFSET,
    CUBE_DROP_MIN_CENTER_HEIGHT,
    CUBE_MAX_LINEAR_SPEED,
    CUBE_MAX_TILT_RADIANS,
    CUBE_WIDTH,
    HOME_GATE_SHARPNESS,
    HOME_REWARD_WEIGHT,
    HOME_SUCCESS_DISTANCE,
    ORIENTATION_REWARD_WEIGHT,
    PUSH_REWARD_WEIGHT,
    REACH_REWARD_WEIGHT,
    ShelfCubeSweepV5EnvCfg,
)
from sweep_rl.shelf_cube_sweep_v5.mdp.rewards import (  # noqa: E402
    pushing_target_raw_reward,
    signed_axis_alignment,
)
from sweep_rl.shelf_cube_sweep_v5.mdp.terminations import (  # noqa: E402
    arm_joint_too_fast,
    cube_below_minimum_height,
    cube_excessive_tilt,
    cube_tilt_angle,
    cube_too_fast,
)

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5"


def _reward(env, name: str) -> torch.Tensor:
    term = env.reward_manager.get_term_cfg(name)
    return term.func(env, **term.params)


def _check_failure_termination_functions(device: str) -> None:
    half_tilt = 0.5
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [torch.cos(torch.tensor(0.6)), 0.0, 0.0, torch.sin(torch.tensor(0.6))],
            [torch.cos(torch.tensor(half_tilt)), torch.sin(torch.tensor(half_tilt)), 0.0, 0.0],
        ],
        device=device,
    )
    angles = cube_tilt_angle(quaternions)
    torch.testing.assert_close(angles[:2], torch.zeros_like(angles[:2]))
    torch.testing.assert_close(angles[2], torch.tensor(1.0, device=device))

    target = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor(
                [[0.0, 0.0, 1.15], [0.0, 0.0, 1.03]], device=device
            ),
            root_quat_w=torch.stack((quaternions[0], quaternions[2])),
            root_lin_vel_w=torch.tensor(
                [[0.0, 0.29, 0.0], [0.0, 0.31, 0.0]], device=device
            ),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_vel=torch.tensor(
                [[0.0, 0.99, 0.0], [0.0, 1.01, 0.0]], device=device
            )
        )
    )
    fake_env = SimpleNamespace(scene={"target_object": target, "robot": robot})
    assert cube_below_minimum_height(
        fake_env, CUBE_DROP_MIN_CENTER_HEIGHT
    ).tolist() == [False, True]
    assert cube_excessive_tilt(fake_env, CUBE_MAX_TILT_RADIANS).tolist() == [
        False,
        True,
    ]
    assert cube_too_fast(fake_env, CUBE_MAX_LINEAR_SPEED).tolist() == [False, True]
    assert arm_joint_too_fast(fake_env, ARM_MAX_JOINT_SPEED).tolist() == [
        False,
        True,
    ]


def _check_random_sweep_reward_math(device: str) -> None:
    axes = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        device=device,
    )
    shelf_z = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand_as(axes)
    torch.testing.assert_close(
        signed_axis_alignment(axes, shelf_z),
        torch.tensor([1.0, 0.0, -1.0], device=device),
    )

    reward = pushing_target_raw_reward(
        distance=torch.tensor([0.0, 0.18, 0.09, 0.09, 0.09], device=device),
        progress_delta=torch.tensor(
            [0.0, 0.0, 0.5, 0.5, 0.5], device=device
        ),
        contact_distance=torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.05], device=device
        ),
        wrist_y_error=torch.zeros(5, device=device),
        target_velocity=torch.tensor(
            [0.0, 0.0, 0.075, 0.11, 0.075], device=device
        ),
        gate_distance=0.04,
    )
    torch.testing.assert_close(
        reward, torch.tensor([2.0, 0.0, 1.0, 0.0, 0.0], device=device)
    )


def main() -> None:
    spec = gym.spec(TASK_ID)
    assert "shelf_cube_sweep_v5.env_cfg" in spec.kwargs["env_cfg_entry_point"]

    cfg = ShelfCubeSweepV5EnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.scene.ground = None
    cfg.scene.shelf.spawn = sim_utils.CuboidCfg(
        size=(0.4, 1.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg.scene.shelf.init_state.pos = (-0.7, 0.0, 1.0)
    cfg.scene.shelf_floor_contact = None
    cfg.rewards.shelf_collision = None

    agent_cfg = ShelfCubeSweepV5PPORunnerCfg()
    _check_failure_termination_functions(agent_cfg.device)
    _check_random_sweep_reward_math(agent_cfg.device)
    assert agent_cfg.actor.hidden_dims == [256, 128, 64]
    assert agent_cfg.critic.hidden_dims == [256, 128, 64]
    assert agent_cfg.num_steps_per_env == 36
    assert agent_cfg.max_iterations == 90_000
    assert agent_cfg.clip_actions is None
    assert agent_cfg.actor.distribution_cfg.init_std == 1.0
    assert agent_cfg.actor.distribution_cfg.std_type == "scalar"
    assert agent_cfg.algorithm.entropy_coef == 0.005
    assert agent_cfg.algorithm.learning_rate == 1.0e-3
    assert agent_cfg.algorithm.schedule == "adaptive"
    assert agent_cfg.algorithm.gamma == 0.98
    assert agent_cfg.algorithm.desired_kl == 0.02
    distribution = GaussianDistribution(
        output_dim=6,
        init_std=agent_cfg.actor.distribution_cfg.init_std,
        std_type=agent_cfg.actor.distribution_cfg.std_type,
    )
    assert distribution.std_param.shape == (6,)
    assert torch.all(distribution.std_param > 0.0)

    env = gym.make(TASK_ID, cfg=cfg)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        assert observations["policy"].shape == (args.num_envs, 39)
        assert unwrapped.action_manager.total_action_dim == 6
        assert unwrapped.action_manager.active_terms == ["arm_action"]
        assert unwrapped.observation_manager.active_terms["policy"] == [
            "joint_pos",
            "joint_vel",
            "pose_command",
            "target_object_position",
            "cube_width",
            "current_ee_pose",
            "sweep_goal_position",
            "actions",
        ]
        assert unwrapped.termination_manager.active_terms == [
            "time_out",
            "cube_drop",
            "cube_tilt",
            "cube_too_fast",
            "arm_joint_too_fast",
        ]
        assert unwrapped.cfg.sim.dt == 0.01
        assert unwrapped.cfg.decimation == 2
        assert unwrapped.max_episode_length == 500

        arm_action = unwrapped.action_manager.get_term("arm_action")
        assert type(arm_action).__name__ == "ArmJointPositionWithOpenGripperAction"
        assert arm_action.action_dim == 6
        assert arm_action.cfg.preserve_order is False
        assert not hasattr(arm_action, "_max_delta")
        robot = unwrapped.scene["robot"]
        cube = unwrapped.scene["target_object"]
        assert cfg.scene.target_object.spawn.center_of_mass == CUBE_COM_OFFSET

        stage = sim_utils.get_current_stage()
        articulation_prims = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith("/World/envs/env_0/Robot")
            and prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
        ]
        assert len(articulation_prims) == 1
        articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prims[0])
        assert articulation_api.GetEnabledSelfCollisionsAttr().Get() is True

        command = unwrapped.command_manager.get_term("ee_pose")
        assert not hasattr(command, "success_latched")
        assert not hasattr(command, "push_approach_latched")
        assert not hasattr(command, "push_approach_alpha")
        assert (
            command.cfg.behind_width_scale == CONTACT_POINT_BEHIND_WIDTH_SCALE
        )
        assert command.cfg.x_offset == CONTACT_POINT_X_OFFSET
        assert command.cfg.z_offset == CONTACT_POINT_Z_OFFSET
        ids = torch.arange(args.num_envs, device=unwrapped.device)
        command._update_target_position(ids)
        contact_offset = command.reach_target_pos_w - cube.data.root_pos_w
        torch.testing.assert_close(
            contact_offset,
            torch.tensor(
                [
                    CONTACT_POINT_X_OFFSET,
                    -0.5 * CUBE_WIDTH,
                    CONTACT_POINT_Z_OFFSET,
                ],
                device=unwrapped.device,
            ).expand_as(contact_offset),
        )
        command.tcp_frame.data.target_pos_w[:, command.cfg.frame_index] = (
            command.reach_target_pos_w
        )
        command.wrist_frame.data.target_pos_w[
            :, command.cfg.wrist_frame_index, 1
        ] = command.reach_target_pos_w[:, 1]
        command._update_metrics()
        assert torch.all(command.metrics["zeta_active"] == 1.0)

        push_cfg = unwrapped.reward_manager.get_term_cfg("pushing_target")
        push_cfg.func.reset()
        cube_state = cube.data.root_state_w.clone()
        cube_state[:, 1] += 0.001
        cube_state[:, 7:] = 0.0
        cube.write_root_state_to_sim(cube_state)
        command._update_target_position(ids)
        command.tcp_frame.data.target_pos_w[:, command.cfg.frame_index] = (
            command.reach_target_pos_w
        )
        command.wrist_frame.data.target_pos_w[
            :, command.cfg.wrist_frame_index, 1
        ] = command.reach_target_pos_w[:, 1]
        delta_push = _reward(unwrapped, "pushing_target")
        expected_delta_push = torch.full_like(
            delta_push, 0.001 / (0.18 * unwrapped.step_dt)
        )
        torch.testing.assert_close(delta_push, expected_delta_push)
        stationary_push = _reward(unwrapped, "pushing_target")
        torch.testing.assert_close(
            stationary_push, torch.zeros_like(stationary_push)
        )

        cube_state = cube.data.root_state_w.clone()
        cube_state[:, :3] = command.goal_pos_w
        cube_state[:, 1] += 0.10
        cube_state[:, 7:] = 0.0
        cube.write_root_state_to_sim(cube_state)
        home_far = _reward(unwrapped, "homing_after_sweep")
        cube_state[:, :3] = command.goal_pos_w
        cube.write_root_state_to_sim(cube_state)
        home_goal = _reward(unwrapped, "homing_after_sweep")
        push_goal = _reward(unwrapped, "pushing_target")
        assert torch.all(home_goal > home_far)
        assert torch.all(push_goal > 0.0)

        home_cfg = unwrapped.reward_manager.get_term_cfg("homing_after_sweep")
        assert type(push_cfg.func).__name__ == "PushingTargetDeltaReward"
        assert home_cfg.weight == HOME_REWARD_WEIGHT
        assert push_cfg.weight == PUSH_REWARD_WEIGHT
        assert push_cfg.weight == 2.0 * REACH_REWARD_WEIGHT
        orientation_cfg = unwrapped.reward_manager.get_term_cfg(
            "end_effector_orientation_tracking"
        )
        assert orientation_cfg.weight == ORIENTATION_REWARD_WEIGHT
        assert orientation_cfg.func.__name__ == "align_ee_to_shelf"
        assert home_cfg.params["success_distance"] == HOME_SUCCESS_DISTANCE
        assert home_cfg.params["gate_sharpness"] == HOME_GATE_SHARPNESS
        assert len(home_cfg.params["asset_cfg"].joint_ids) == 5
        assert unwrapped.reward_manager.get_term_cfg("action_rate").weight == -0.03
        assert unwrapped.reward_manager.get_term_cfg("joint_vel").weight == -0.03

        test_actions = (
            torch.full((args.num_envs, 6), -2.0, device=unwrapped.device),
            torch.full((args.num_envs, 6), 2.0, device=unwrapped.device),
            torch.zeros((args.num_envs, 6), device=unwrapped.device),
        )
        for action in test_actions:
            observations, rewards, terminated, _, _ = env.step(action)
            gripper_targets = robot.data.joint_pos_target[
                :, arm_action._gripper_joint_ids
            ]
            torch.testing.assert_close(
                gripper_targets, torch.zeros_like(gripper_targets)
            )
            assert observations["policy"].shape == (args.num_envs, 39)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(arm_action.processed_actions).all()
        for metric_name in (
            "goal_xyz_distance",
            "goal_yz_distance",
            "cube_y_displacement",
            "cube_y_delta",
            "cube_y_velocity",
            "normalized_progress_rate",
            "tcp_contact_distance",
            "wrist_y_error",
            "zeta_active",
            "home_gate",
            "home_joint_error_5",
        ):
            assert metric_name in command.metrics
            assert torch.isfinite(command.metrics[metric_name]).all()
        print("standalone shelf_cube_sweep_v5 smoke test: passed", flush=True)
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
