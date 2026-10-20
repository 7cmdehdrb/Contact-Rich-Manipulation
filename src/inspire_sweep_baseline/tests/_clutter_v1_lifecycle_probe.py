"""Explicit Isaac/PhysX probe of v1 observation, reward, and reset contracts."""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.assets.tactile_regions import TACTILE_REGIONS, parent_sensor_name
from inspire_sweep_baseline.contact_sweep_clutter_v1.mdp import (
    contact_any, contact_count, goal_hold_xy, obstacle_displacement_l2,
)
from inspire_sweep_baseline.mdp.observations import object_position_in_root


task = "Isaac-Inspire-Contact-Sweep-Clutter-v1"
try:
    with torch.inference_mode():
        assert args.num_envs >= 4
        cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        parent = parse_env_cfg(
            "Isaac-Inspire-Contact-Sweep-Clutter-v0",
            device=cfg.sim.device, num_envs=args.num_envs,
        )
        assert parent.rewards.goal_distance.weight == 94.0
        assert parent.rewards.pushing_speed.params["free_speed"] == 0.2
        assert parent.rewards.arm_speed_excess.params["free_speed"] == 1.0
        assert cfg.rewards.reaching.weight == 30.0
        assert cfg.rewards.goal_distance.weight == 100.0
        assert cfg.rewards.pushing_speed.params["free_speed"] == 1.0
        assert cfg.rewards.arm_speed_excess.params["free_speed"] == 0.5
        for name in ("arm_action_rate", "hand_action_rate", "arm_joint_velocity", "hand_joint_velocity"):
            assert getattr(cfg.rewards, name).to_dict() == getattr(parent.rewards, name).to_dict()
        assert cfg.terminations.to_dict() == parent.terminations.to_dict()
        assert cfg.events.to_dict() == parent.events.to_dict()
        cfg.seed = 42
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset(seed=42)
            shapes = {name: list(value.shape) for name, value in obs.items()}
            assert shapes == {
                "policy": [env.num_envs, 57], "obstacles": [env.num_envs, 40],
                "privileged": [env.num_envs, 3],
            }
            objects = env.scene["object_collection"]
            initial = env._initial_object_positions_w.clone()
            assert torch.equal(initial, objects.data.object_pos_w)
            actor_initial = obs["policy"][:, 20:23].clone()
            initial_noise = actor_initial - obs["privileged"]
            assert initial_noise.abs().max() <= 0.010001
            assert torch.equal(obs["privileged"], object_position_in_root(env))
            obstacle_observation = obs["obstacles"].clone()

            # Move only the target: actor's initial reading and initial geometry
            # remain fixed, while privileged XYZ follows the new physical state.
            state = objects.data.object_link_state_w.clone()
            state[:, 0, 0] += 0.07
            objects.write_object_link_state_to_sim(state)
            changed = env.observation_manager.compute()
            assert torch.equal(changed["policy"][:, 20:23], actor_initial)
            assert torch.equal(changed["obstacles"], obstacle_observation)
            assert not torch.equal(changed["privileged"], obs["privileged"])
            assert torch.equal(changed["privileged"], object_position_in_root(env))
            assert torch.equal(obstacle_displacement_l2(env), torch.zeros(env.num_envs, device=env.device))

            # World XY gate must ignore arbitrary target Z.
            goal = env.command_manager.get_command("target_goal_pos")
            state[:, 0, :3] = goal[:, :3]
            state[:, 0, 2] += 0.3
            objects.write_object_link_state_to_sim(state)
            assert torch.all(goal_hold_xy(env) == 1)
            state[:, 0, 0] += 0.0101
            objects.write_object_link_state_to_sim(state)
            assert torch.all(goal_hold_xy(env) == 0)

            # Two obstacles 10 cm from their true initial XYZ cost .02 m^2;
            # z-only movement counts, while noisy observations never enter cost.
            state[:, :, :3] = initial
            state[:, 0, :3] += 2.0
            state[:, 1, 0] += 0.1
            state[:, 2, 2] += 0.1
            objects.write_object_link_state_to_sim(state)
            env._initial_obstacle_observation += 100.0
            measured_cost = obstacle_displacement_l2(env)
            torch.testing.assert_close(measured_cost, torch.full_like(measured_cost, 0.02))
            state[:, 3, 1] += 10.0
            objects.write_object_link_state_to_sim(state)
            torch.testing.assert_close(
                obstacle_displacement_l2(env), torch.full_like(measured_cost, 0.04),
            )

            # A partial reset replaces only selected episode snapshots.
            obs, _ = env.reset(seed=43)
            before_actor = obs["policy"][:, 20:23].clone()
            before_geometry = env._initial_object_positions_w.clone()
            before_obstacles = obs["obstacles"].clone()
            ids = torch.tensor([1, 3], device=env.device)
            obs, _ = env.reset(env_ids=ids)
            assert torch.equal(obs["policy"][[0, 2], 20:23], before_actor[[0, 2]])
            assert torch.equal(env._initial_object_positions_w[[0, 2]], before_geometry[[0, 2]])
            assert torch.equal(obs["obstacles"][[0, 2]], before_obstacles[[0, 2]])
            assert not torch.equal(env._initial_object_positions_w[ids], before_geometry[ids])
            assert torch.equal(env._initial_object_positions_w[ids], objects.data.object_pos_w[ids])
            assert (obs["policy"][ids, 20:23] - obs["privileged"][ids]).abs().max() <= 0.010001

            # Normal physics transitions use fresh contacts for both reward
            # terms; post-reset observations mask terminated rows.
            action = torch.zeros(env.num_envs, 8, device=env.device)
            action[:, -2:] = 1.0
            contact_frames = 0
            for _ in range(20):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                count = (obs["policy"][:, 34:51] > 0.5).sum(dim=-1).float()
                torch.testing.assert_close(contact_any(env), (count > 0).float())
                torch.testing.assert_close(contact_count(env), count.clamp(max=5) / 5)
                contact_frames += int((count > 0).sum())
                active = ~(terminated | truncated)
                torch.testing.assert_close(
                    env.reward_manager._step_reward[active, env.reward_manager.active_terms.index("contact_any")],
                    0.16 * (count[active] > 0).float(),
                )

            env.episode_length_buf[ids] = env.max_episode_length - 1
            obs, _, _, truncated, _ = env.step(action)
            assert truncated[ids].all()
            assert torch.equal(env._initial_object_positions_w[ids], objects.data.object_pos_w[ids])
            assert (obs["policy"][ids, 20:23] - obs["privileged"][ids]).abs().max() <= 0.010001
            assert torch.count_nonzero(obs["policy"][ids, 34:51]) == 0
            assert torch.count_nonzero(contact_any(env)[ids]) == 0

            # Deliberately place a real obstacle against the calibrated palm
            # region, using the existing v0 contact probe's fixture geometry.
            env.reset(seed=42)
            robot = env.scene["robot"]
            region = TACTILE_REGIONS[0]
            pad_id = robot.body_names.index(region["name"])
            sensor = env.scene[parent_sensor_name(region["parent_name"])]
            tactile = env.observation_manager.cfg.policy.tactile
            fixed_joints = robot.data.joint_pos.clone()
            positive_contact = None
            offsets = [
                (0., 0.), (0.02, 0.), (-0.02, 0.), (0.04, 0.), (-0.04, 0.),
                (0., 0.02), (0., -0.02), (0.02, 0.02), (-0.02, -0.02),
            ]
            for offset_x, offset_y in offsets:
                robot.write_joint_state_to_sim(fixed_joints, torch.zeros_like(robot.data.joint_vel))
                state = objects.data.object_link_state_w.clone()
                state[..., :3] = env.scene.env_origins[:, None] + state.new_tensor([0., 2., 4.])
                state[..., 0] += torch.arange(objects.num_objects, device=env.device)[None] * 0.15
                state[..., 3:7] = state.new_tensor([1., 0., 0., 0.])
                state[..., 7:] = 0.
                pad_pos = robot.data.body_link_pos_w[0, pad_id]
                pad_quat = robot.data.body_link_quat_w[0, pad_id]
                point = state.new_tensor(region["verification"]["parent_surface_point_local_m"])
                normal = state.new_tensor(region["verification"]["parent_surface_outward_normal_local"])
                world_normal = quat_apply(pad_quat, normal)
                state[0, 1, :3] = (
                    pad_pos + quat_apply(pad_quat, point)
                    + quat_apply(pad_quat, point.new_tensor([offset_x, offset_y, 0.]))
                    - 0.002 * world_normal
                )
                rotation = torch.cat((
                    (1. + world_normal[2]).reshape(1),
                    torch.linalg.cross(world_normal.new_tensor([0., 0., 1.]), world_normal),
                ))
                state[0, 1, 3:7] = rotation / torch.linalg.vector_norm(rotation)
                objects.write_object_link_state_to_sim(state)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                env.sensor_data_fresh[:] = True
                # This deliberately bypasses env.step and its control counter.
                tactile.func.reset()
                bits = tactile.func(env, **tactile.params)
                raw = sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
                assert raw[4][0, 0] == 0, "Target must not explain the obstacle contact"
                if raw[4][0, 1] > 0 and bits[0, 0] > 0:
                    env.reward_manager.compute(env.step_dt)
                    on_count = int((bits[0] > 0.5).sum())
                    any_index = env.reward_manager.active_terms.index("contact_any")
                    count_index = env.reward_manager.active_terms.index("contact_count")
                    rates = env.reward_manager._step_reward[0, [any_index, count_index]]
                    expected = rates.new_tensor([0.16, 0.16 * min(on_count, 5) / 5])
                    torch.testing.assert_close(rates, expected)
                    positive_contact = {
                        "active_channels": on_count,
                        "weighted_contact_rates": rates.tolist(),
                        "obstacle_parent_contact_points": int(raw[4][0, 1]),
                    }
                    break
            assert positive_contact is not None, "Controlled obstacle palm contact was not observed"

            report = {
                "task": task, "seed": 42, "num_envs": env.num_envs,
                "observation_shapes": shapes,
                "initial_target_noise_max_m": float(initial_noise.abs().max()),
                "reward_terms": env.reward_manager.active_terms,
                "two_obstacles_10cm_cost_m2": measured_cost.tolist(),
                "two_obstacles_10cm_penalty_over_10s": -5.0 * float(measured_cost[0]) * 10.0,
                "max_obstacle_penalty_per_second": -0.2,
                "normal_physics_steps": 20, "contact_environment_steps": contact_frames,
                "controlled_positive_contact": positive_contact,
                "checks": [
                    "v0_config_unchanged", "four_motion_penalties_inherited",
                    "actor_initial_target_fixed", "critic_target_live",
                    "world_xy_hold_ignores_height", "target_excluded_from_displacement",
                    "obstacle_3d_cost_uses_noiseless_snapshot", "obstacle_cap",
                    "partial_reset", "fresh_contact_rewards", "timeout_auto_reset",
                    "real_obstacle_contact_positive_rewards",
                ],
                "scope": "Manipulated-state contract probe; not learned-policy performance.",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            env.close()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    launcher.app.close()
