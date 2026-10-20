"""Explicit PhysX validation of Clutter v2 compensated six-axis wrench.

The free-space checks move the shelf and objects away and bypass task
terminations. Known forces and moments are applied to the real articulation,
not injected into the sensor tensor. This is a sensor/lifecycle contract probe,
not a learned-policy performance evaluation.
"""

import argparse
import json
import math
import os
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
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.contact_sweep_clutter_v2.wrench import PureWrenchReader


def stats(values):
    values = torch.stack(values).detach()
    return {
        "sample_count": values.shape[0],
        "force_rms_N": float(values[..., :3].square().mean().sqrt()),
        "force_max_norm_N": float(torch.linalg.vector_norm(values[..., :3], dim=-1).max()),
        "moment_rms_Nm": float(values[..., 3:].square().mean().sqrt()),
        "moment_max_norm_Nm": float(torch.linalg.vector_norm(values[..., 3:], dim=-1).max()),
        "mean_per_env": values.mean(dim=0).cpu().tolist(),
    }


task = "Isaac-Inspire-Contact-Sweep-Clutter-v2"
report = {
    "task": task,
    "checks": [],
    "scope": "Manipulated-state PhysX sensor and reset checks; no trained-policy claims.",
}
try:
    with torch.inference_mode():
        assert args.num_envs == 4, "Known-load fixture uses exactly four environments."
        cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        parent_cfg = parse_env_cfg(
            "Isaac-Inspire-Contact-Sweep-Clutter-v1", device=cfg.sim.device, num_envs=args.num_envs,
        )
        parent_rewards = parent_cfg.rewards.to_dict()
        current_rewards = cfg.rewards.to_dict()
        assert set(current_rewards) - set(parent_rewards) == {"wrench_change"}
        for name, value in parent_rewards.items():
            assert current_rewards[name] == value, f"Inherited reward changed: {name}"
        for name in ("actions", "events", "terminations", "commands", "scene"):
            assert getattr(cfg, name).to_dict() == getattr(parent_cfg, name).to_dict(), name
        parent_observations = parent_cfg.observations.to_dict()
        current_observations = cfg.observations.to_dict()
        current_observations["policy"]["wrist_wrench"] = parent_observations["policy"]["wrist_wrench"]
        assert current_observations == parent_observations
        report["checks"].append("v1_rewards_actions_events_terminations_commands_scene_preserved")
        report["checks"].append("only_wrist_observation_replaced")
        cfg.seed = 42
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            observation, _ = env.reset(seed=42)
            robot = env.scene["robot"]
            reader = PureWrenchReader(robot, gravity_w=cfg.sim.gravity)
            report["observation_shapes"] = {key: list(value.shape) for key, value in observation.items()}
            assert report["observation_shapes"] == {
                "policy": [4, 57], "obstacles": [4, 40], "privileged": [4, 3],
            }
            assert torch.count_nonzero(observation["policy"][:, -6:]) == 0
            report["checks"].append("reset_wrench_zero")

            # The normal task shelf would touch the hand in some trajectories.
            # This fixture isolates the downstream Newton--Euler balance.
            shelf = env.scene["shelf"]
            shelf_state = shelf.data.root_state_w.clone()
            shelf_state[:, :3] = env.scene.env_origins + shelf_state.new_tensor([0., 1000., 0.])
            shelf_state[:, 7:] = 0.
            shelf.write_root_state_to_sim(shelf_state)
            objects = env.scene["object_collection"]
            object_state = objects.data.object_link_state_w.clone()
            object_state[..., :3] = (
                env.scene.env_origins[:, None] + object_state.new_tensor([0., 2000., 2.])
            )
            object_state[..., 0] += torch.arange(objects.num_objects, device=env.device)[None] * 0.2
            object_state[..., 7:] = 0.
            objects.write_object_link_state_to_sim(object_state)

            action = torch.zeros((env.num_envs, 8), device=env.device)
            action[:, 6:] = 1.

            def advance(command, external_force=None, external_moment=None):
                env.action_manager.process_action(command)
                for _ in range(cfg.decimation):
                    env._sim_step_counter += 1
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    if external_force is not None:
                        # Bypass WrenchComposer's version-specific position
                        # handling and send the known world wrench directly
                        # to PhysX, independently of the sensor calculation.
                        forces = torch.zeros((env.num_envs, robot.num_bodies, 3), device=env.device)
                        moments = torch.zeros_like(forces)
                        forces[:, child_id] = external_force
                        moments[:, child_id] = external_moment
                        positions = robot.data.body_com_pos_w.clone()
                        advance.last_application_pos = positions[:, child_id].clone()
                        robot.root_physx_view.apply_forces_and_torques_at_position(
                            force_data=forces.reshape(-1, 3),
                            torque_data=moments.reshape(-1, 3),
                            position_data=positions.reshape(-1, 3),
                            indices=robot._ALL_INDICES,
                            is_global=True,
                        )
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                env.common_step_counter += 1
                env.episode_length_buf += 1
                env.sensor_data_fresh[:] = True
                frame = env.scene["ee_frame"].data
                result = reader.read(
                    frame.target_pos_w[:, 0], frame.target_quat_w[:, 0],
                    valid_env_mask=env.sensor_data_fresh,
                )
                assert torch.isfinite(result.pure_eef).all()
                return result

            for _ in range(50):
                advance(action)
            static_raw, static_pure = [], []
            for _ in range(40):
                sample = advance(action)
                static_raw.append(sample.raw_world.clone())
                static_pure.append(sample.pure_world.clone())
            report["free_static_raw"] = stats(static_raw)
            report["free_static_pure"] = stats(static_pure)

            moving_raw, moving_pure = [], []
            arm_speed, hand_speed = [], []
            initial_joints = robot.data.joint_pos.clone()
            for index in range(160):
                phase = index * env.step_dt * 2. * math.pi * 0.6
                action[:, 0] = 0.12 * math.sin(phase)
                action[:, 1] = 0.10 * math.sin(phase)
                action[:, 2] = -0.10 * math.sin(phase)
                action[:, 3] = 0.15 * math.sin(phase)
                action[:, 5] = -0.15 * math.sin(phase)
                action[:, 6] = 0.55 + 0.40 * math.cos(phase)
                action[:, 7] = 0.60 + 0.30 * math.cos(phase)
                sample = advance(action)
                moving_raw.append(sample.raw_world.clone())
                moving_pure.append(sample.pure_world.clone())
                arm_speed.append(robot.data.joint_vel[:, :6].abs().max())
                hand_speed.append(robot.data.joint_vel[:, 6:].abs().max())
            report["moving_raw"] = stats(moving_raw)
            report["moving_pure"] = stats(moving_pure)
            report["moving_joint_max_change_rad"] = float((robot.data.joint_pos - initial_joints).abs().max())
            report["moving_max_arm_speed_rad_s"] = float(torch.stack(arm_speed).max())
            report["moving_max_hand_speed_rad_s"] = float(torch.stack(hand_speed).max())

            action[:] = 0.
            action[:, 6:] = 1.
            for _ in range(60):
                advance(action)

            # Independent six-axis stimulus at the actual hand-base COM.
            # Pure output is the parent-to-child support reaction, opposite
            # the externally applied wrench on the downstream hand.
            child_id = robot.body_names.index("inspire_base_link")
            force = torch.tensor(
                [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.], [2., -3., 4.]],
                device=env.device,
            )
            moment = torch.tensor(
                [[0., .20, 0.], [.15, 0., 0.], [0., 0., .18], [.10, -.12, .08]],
                device=env.device,
            )
            loaded_pure, loaded_expected, loaded_error = [], [], []
            for index in range(70):
                # An explicit world application point removes ambiguity about
                # whether the simulator defaults to the link origin or COM.
                sample = advance(action, force, moment)
                application_pos = advance.last_application_pos
                eef_pos = env.scene["ee_frame"].data.target_pos_w[:, 0]
                expected = -torch.cat((
                    force,
                    moment + torch.cross(application_pos - eef_pos, force, dim=-1),
                ), dim=-1)
                if index >= 30:
                    loaded_pure.append(sample.pure_world.clone())
                    loaded_expected.append(expected)
                    loaded_error.append(sample.pure_world - expected)
            report["known_external_pure"] = stats(loaded_pure)
            report["known_external_expected"] = stats(loaded_expected)
            report["known_external_error"] = stats(loaded_error)
            # Return to the unmodified task to test observation freshness and
            # reward lifecycle through the actual public env.step/reset path.
            observation, _ = env.reset(seed=43)
            penalty_index = env.reward_manager.active_terms.index("wrench_change")
            max_observed_penalty_rate = 0.0
            for index in range(4):
                observation, reward, terminated, timeout, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in observation.values())
                assert torch.count_nonzero(observation["policy"][terminated | timeout, -6:]) == 0
                rates = env.reward_manager._step_reward[:, penalty_index]
                assert torch.all(rates <= 0.)
                assert torch.all(rates >= -0.100001)
                max_observed_penalty_rate = max(max_observed_penalty_rate, float((-rates).max()))
                if index == 0:
                    assert torch.count_nonzero(rates) == 0
                positive_earned = torch.zeros(env.num_envs, device=env.device)
                for name in env.reward_manager.active_terms:
                    if env.reward_manager.get_term_cfg(name).weight > 0.:
                        positive_earned += env.reward_manager._episode_sums[name].clamp_min(0.)
                penalty_spent = -env.reward_manager._episode_sums["wrench_change"]
                assert torch.all(penalty_spent <= 0.05 * positive_earned + 1e-6)
                cached = env.observation_manager.cfg.policy.wrist_wrench.func(env)
                torch.testing.assert_close(observation["policy"][:, -6:], cached)
            before = observation["policy"][:, -6:].clone()
            ids = torch.tensor([1, 3], device=env.device)
            observation, _ = env.reset(env_ids=ids)
            assert torch.count_nonzero(observation["policy"][ids, -6:]) == 0
            torch.testing.assert_close(observation["policy"][[0, 2], -6:], before[[0, 2]])
            report["maximum_observed_penalty_rate"] = max_observed_penalty_rate
            report["checks"].extend([
                "finite_normal_steps", "partial_reset_wrench_zero",
                "partial_reset_preserves_other_rows", "shared_observation_reward_sample",
                "first_fresh_sample_no_change_penalty",
                "penalty_absolute_cap", "penalty_prefix_positive_reward_budget",
            ])

            env.episode_length_buf[ids] = env.max_episode_length - 1
            observation, reward, terminated, timeout, _ = env.step(action)
            assert timeout[ids].all()
            assert torch.count_nonzero(observation["policy"][ids, -6:]) == 0
            assert torch.count_nonzero(env.reward_manager._step_reward[ids, penalty_index]) == 0
            report["checks"].append("automatic_reset_wrench_zero")

            # Record all quantitative evidence even if a physical accuracy
            # assertion fails; tightening/loosening tolerance must be explicit.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
            assert report["free_static_raw"]["force_max_norm_N"] > 0.5
            assert report["free_static_pure"]["force_max_norm_N"] < 0.15
            assert report["free_static_pure"]["moment_max_norm_Nm"] < 0.02
            assert report["moving_joint_max_change_rad"] > 0.05
            assert report["moving_pure"]["force_max_norm_N"] < 0.5
            assert report["moving_pure"]["moment_max_norm_Nm"] < 0.05
            assert report["known_external_error"]["force_max_norm_N"] < 0.15
            assert report["known_external_error"]["moment_max_norm_Nm"] < 0.02
            report["checks"].extend([
                "static_gravity_cancellation",
                "moving_arm_hand_inertial_cancellation",
                "known_external_force_and_moment_sign_origin",
            ])
            report["status"] = "passed"
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.close()
except BaseException:
    report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # The environment is closed by the inner finally above. Kit's default
    # fast shutdown would overwrite a failure with exit status zero, while
    # graceful shutdown segfaults in this installed build. Preserve the saved
    # failure evidence and an unambiguous nonzero status without closing Kit.
    os._exit(1)
finally:
    launcher.app.close()
