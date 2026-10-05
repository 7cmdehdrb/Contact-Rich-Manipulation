#!/usr/bin/env python3
"""Controlled live contact probes; these are sensor checks, not learned holding."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys
import traceback

package_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--support-steps", type=int, default=45)
parser.add_argument("--contact-steps", type=int, default=15)
parser.add_argument("--push-force", type=float, default=0.5)
parser.add_argument("--consecutive-on", type=int, default=2)
# AppLauncher's --verbose also enables the fixture's force/pose diagnostics.
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import hand_manipulation_test
    from hand_manipulation_test.assets.robot import ROBOT_CONTACT_BODY_NAMES
    from hand_manipulation_test.contact_sensors import cube_palm_contact_mask, table_contact_mask
    from hand_manipulation_test.geometry import palm_tactile_bits
    from hand_manipulation_test.sensors import PALM_CHANNEL_NAMES
    from hand_manipulation_test.mdp.contact_rewards import palm_cube_contact_reward
    from hand_manipulation_test.mdp.contact_terminations import contact_time_out, table_contact_failure

    assert args.support_steps > 0 and 0 < args.contact_steps <= 15
    assert args.push_force > 0 and args.consecutive_on >= 2
    task = hand_manipulation_test.CONTACT_TASK_ID
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        observation, _ = env.reset(seed=args.seed)
        assert observation["policy"].shape == (2, 55)
        robot = env.scene["robot"]
        cube = env.scene["target_object"]
        table = env.scene["table"]
        safe_reset = env.event_manager.get_term_cfg("safe_hand").func
        assert bool(safe_reset.check_final().all())
        pinned_q = robot.data.joint_pos.clone()
        zero_joint = torch.zeros_like(pinned_q)
        threshold = env.cfg.task.contact_threshold_n
        initial_target = env.command_manager.get_term("target_position").target_pos_w.clone()
        central_filter = next(
            index for index, path in enumerate(env.cfg.scene.cube_palm_contacts.filter_prim_paths_expr)
            if path.endswith("/inspire_palm_force_sensor")
        )
        assert central_filter == 0
        # Direct physics probes bypass policy counters, so explicitly mark the
        # contact buffers live. No termination/reset manager runs between steps.
        env.episode_length_buf[:] = 1
        gravity = pinned_q.new_tensor(env.cfg.sim.gravity)
        gravity_compensation = -env.cfg.task.cube_mass * gravity
        commanded_cube_force_w = torch.zeros((2, 3), device=env.device)
        pad_ids, pad_names = robot.find_bodies("inspire_palm_force_sensor", preserve_order=True)
        assert tuple(pad_names) == ("inspire_palm_force_sensor",) and len(pad_ids) == 1
        pad_id = int(pad_ids[0])
        pad_local_center = safe_reset.collision_bounds.centers[pad_id].clone()
        pad_half_extents = safe_reset.collision_bounds.half_extents[pad_id].clone()
        corner_signs = pinned_q.new_tensor(tuple(product((-1.0, 1.0), repeat=3)))
        pad_local_corners = pad_local_center + corner_signs * pad_half_extents
        raw_palm_indices = torch.tensor(
            [env.scene["palm_tactile"].body_names.index(name) for name in PALM_CHANNEL_NAMES],
            device=env.device,
        )

        def finite(name, tensor):
            assert bool(torch.isfinite(tensor).all()), f"Non-finite {name}"

        def write_pose(asset, position, quaternion=None):
            state = asset.data.default_root_state.clone()
            state[:, :3] = position
            if quaternion is not None:
                state[:, 3:7] = quaternion
            state[:, 7:] = 0.0
            asset.write_root_state_to_sim(state)

        def cube_force(force):
            nonlocal commanded_cube_force_w
            commanded_cube_force_w = force.reshape(env.num_envs, 3).clone()

        def pin_robot():
            robot.write_joint_state_to_sim(pinned_q, zero_joint)
            robot.set_joint_position_target(pinned_q)
            robot.set_joint_velocity_target(zero_joint)
            robot.set_joint_effort_target(zero_joint)

        def palm_geometry():
            reference, hand_quaternion = safe_reset.palm_reference_pose_w()
            hand_position = robot.data.body_link_pos_w[:, safe_reset.hand_body_id]
            pad_position = robot.data.body_link_pos_w[:, pad_id]
            pad_quaternion = robot.data.body_link_quat_w[:, pad_id]
            normal = quat_apply(hand_quaternion, hand_quaternion.new_tensor((0.0, 1.0, 0.0)).expand(2, -1))
            pad_center_w = pad_position + quat_apply(pad_quaternion, pad_local_center.expand(2, -1))
            world_corners = pad_position[:, None] + quat_apply(
                pad_quaternion[:, None].expand(2, 8, 4).reshape(-1, 4),
                pad_local_corners[None].expand(2, 8, 3).reshape(-1, 3),
            ).reshape(2, 8, 3)
            corners_h = quat_apply_inverse(
                hand_quaternion[:, None].expand(2, 8, 4).reshape(-1, 4),
                (world_corners - hand_position[:, None]).reshape(-1, 3),
            ).reshape(2, 8, 3)
            minimum, maximum = corners_h.amin(1), corners_h.amax(1)
            live_front_h = (minimum + maximum) * 0.5
            live_front_h[:, 1] = maximum[:, 1]
            live_front_w = hand_position + quat_apply(hand_quaternion, live_front_h)
            return {
                "reference": reference.clone(), "hand_position": hand_position.clone(),
                "hand_quaternion": hand_quaternion.clone(), "normal": normal,
                "pad_position": pad_position.clone(), "pad_quaternion": pad_quaternion.clone(),
                "pad_center_w": pad_center_w, "live_front_h": live_front_h,
                "live_front_w": live_front_w,
            }

        def debug_sample(label, geometry=None):
            geometry = palm_geometry() if geometry is None else geometry
            raw = env.scene["palm_tactile"].data.net_forces_w.index_select(1, raw_palm_indices)
            filtered = env.scene["cube_palm_contacts"].data.force_matrix_w[:, 0]
            normal = geometry["normal"]
            delta = cube.data.root_pos_w - geometry["reference"]
            actual_force_w = quat_apply(
                cube.data.root_quat_w,
                cube.permanent_wrench_composer.composed_force_as_torch[:, 0],
            )
            diagnostics = {
                "label": label,
                "raw_palm_17_forces_w": raw.tolist(),
                "raw_palm_17_magnitudes_n": torch.linalg.vector_norm(raw, dim=-1).tolist(),
                "filtered_cube_17_forces_w": filtered.tolist(),
                "filtered_cube_17_magnitudes_n": torch.linalg.vector_norm(filtered, dim=-1).tolist(),
                "cube_net_contact_n": torch.linalg.vector_norm(env.scene["cube_palm_contacts"].data.net_forces_w[:, 0], dim=-1).tolist(),
                "cube_position_w": cube.data.root_pos_w.tolist(), "cube_quaternion_w": cube.data.root_quat_w.tolist(),
                "cube_velocity_w": cube.data.root_state_w[:, 7:13].tolist(),
                "cube_minus_reference_w": delta.tolist(), "cube_normal_distance_m": (delta * normal).sum(-1).tolist(),
                "palm_reference_w": geometry["reference"].tolist(), "palm_reference_h_cached": safe_reset.palm_reference_h.tolist(),
                "hand_position_w": geometry["hand_position"].tolist(), "hand_quaternion_w": geometry["hand_quaternion"].tolist(),
                "normal_w": normal.tolist(), "pad_position_w": geometry["pad_position"].tolist(),
                "pad_quaternion_w": geometry["pad_quaternion"].tolist(),
                "pad_collider_center_body": pad_local_center.tolist(), "pad_collider_half_extents_body": pad_half_extents.tolist(),
                "pad_collider_center_w": geometry["pad_center_w"].tolist(),
                "live_body_bounds_front_w": geometry["live_front_w"].tolist(),
                "live_body_bounds_front_h": geometry["live_front_h"].tolist(),
                "live_front_minus_cached_reference_m": (geometry["live_front_w"] - geometry["reference"]).tolist(),
                "commanded_external_force_w": commanded_cube_force_w.tolist(), "actual_external_force_w": actual_force_w.tolist(),
                "robot_q_minus_pin_max": (robot.data.joint_pos - pinned_q).abs().amax(-1).tolist(),
                "contact_reward": palm_cube_contact_reward(env).tolist(),
                "table_failure": table_contact_failure(env).tolist(),
            }
            if args.verbose:
                print("CONTACT_FIXTURE_DEBUG", diagnostics, flush=True)

        def physics_step():
            # Impose only the robot state, never the contact/sensor tensors.
            # This keeps the sensor fixture stationary without a policy or IK
            # operation while PhysX resolves the Cube contact itself.
            pin_robot()
            # The permanent composer caches link poses until reset. Rebuild
            # the intended WORLD force before each step using the live Cube
            # pose; retaining a local force after rotating the Cube is wrong.
            composer = cube.permanent_wrench_composer
            composer.reset()
            force = commanded_cube_force_w[:, None].contiguous()
            composer.set_forces_and_torques(
                forces=force, torques=torch.zeros_like(force), is_global=True,
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env._sim_step_counter += 1
            finite("robot joints", robot.data.joint_pos)
            finite("Cube state", cube.data.root_state_w)
            finite("Cube-palm contacts", env.scene["cube_palm_contacts"].data.force_matrix_w)
            finite("table contact history", env.scene["table_contacts"].data.force_matrix_w_history)

        def central_force():
            matrix = env.scene["cube_palm_contacts"].data.force_matrix_w
            assert matrix.shape == (2, 1, 17, 3), matrix.shape
            return torch.linalg.vector_norm(matrix[:, 0, central_filter], dim=-1)

        # First let the Cube settle normally under gravity on the real table.
        cube_force(torch.zeros((2, 3), device=env.device))
        support_peak = torch.zeros(2, device=env.device)
        for _ in range(args.support_steps):
            physics_step()
            net_table = env.scene["table_contacts"].data.net_forces_w
            support_peak = torch.maximum(support_peak, torch.linalg.vector_norm(net_table[:, 0], dim=-1))
            assert not bool(table_contact_mask(env).any()), "Cube support was misclassified as robot-table contact"
        assert bool((support_peak > 1.0).all()), f"No physical Cube support load: {support_peak.tolist()}"
        print("SUPPORT_ONLY", {"raw_table_peak_n": support_peak.tolist(),
                               "robot_table_failure": table_contact_failure(env).tolist()}, flush=True)
        if args.verbose:
            print("CONTACT_SENSOR_METADATA", {"canonical_palm_order": PALM_CHANNEL_NAMES,
                  "native_palm_order": env.scene["palm_tactile"].body_names,
                  "cube_filter_paths": env.cfg.scene.cube_palm_contacts.filter_prim_paths_expr,
                  "central_robot_body_id": pad_id}, flush=True)

        # Remove the table from the palm-contact fixture so its failure mask
        # cannot suppress the otherwise valid Cube-palm reward.
        far_table = table.data.root_pos_w.clone()
        far_table[:, 2] -= 5.0
        write_pose(table, far_table)
        cube_force(gravity_compensation.expand(2, -1).clone())
        pin_robot()
        palm_position, hand_quaternion = safe_reset.palm_reference_pose_w()
        palm_position = palm_position.clone()
        hand_quaternion = hand_quaternion.clone()
        normal = quat_apply(hand_quaternion, hand_quaternion.new_tensor((0.0, 1.0, 0.0)).expand(2, -1))
        far_cube = palm_position + 0.30 * normal
        far_cube[:, 2] += 0.40
        write_pose(cube, far_cube, hand_quaternion)
        for _ in range(5):
            physics_step()
        assert not bool(table_contact_mask(env).any())
        assert not bool(cube_palm_contact_mask(env).any())

        peak_central = torch.zeros(2, device=env.device)
        peak_raw_pads = torch.zeros((2, 17), device=env.device)
        peak_filtered_pads = torch.zeros((2, 17), device=env.device)
        best_streak = torch.zeros(2, dtype=torch.long, device=env.device)
        accepted_gap = None
        # Shallow physical overlap supplies a reliable initial loaded contact.
        # The Cube thereafter evolves dynamically; it is not re-teleported on
        # each substep. No synthetic tactile bits or forces enter the sensor.
        trials = ((source, penetration) for source in ("cached_H", "live_body_bounds")
                  for penetration in (0.0005, 0.0010, 0.0020, 0.0030))
        for source, penetration_m in trials:
            write_pose(cube, far_cube, hand_quaternion)
            cube_force(gravity_compensation.expand(2, -1).clone())
            for _ in range(4):
                physics_step()
            # Compare the post-step pose with the pose actually imposed on
            # the next step. Place the Cube only AFTER pinning and lazy FK.
            pre_pin_geometry = palm_geometry()
            pin_robot()
            geometry = palm_geometry()
            if args.verbose:
                print("PIN_POSE_DELTA", {"source": source, "penetration_m": penetration_m,
                      "reference_shift_w": (geometry["reference"] - pre_pin_geometry["reference"]).tolist(),
                      "pad_body_shift_w": (geometry["pad_position"] - pre_pin_geometry["pad_position"]).tolist()}, flush=True)
            palm_position = geometry["reference"] if source == "cached_H" else geometry["live_front_w"]
            hand_quaternion, normal = geometry["hand_quaternion"], geometry["normal"]
            contact_center = palm_position + (0.5 * env.cfg.task.cube_size - penetration_m) * normal
            write_pose(cube, contact_center, hand_quaternion)
            cube_force(gravity_compensation.expand(2, -1) - args.push_force * normal)
            debug_sample(f"{source}:{penetration_m}:placed", geometry)
            streak = torch.zeros(2, dtype=torch.long, device=env.device)
            for step in range(args.contact_steps):
                physics_step()
                load = central_force()
                peak_central = torch.maximum(peak_central, load)
                bits = palm_tactile_bits(env)
                reward = palm_cube_contact_reward(env)
                raw = env.scene["palm_tactile"].data.net_forces_w.index_select(1, raw_palm_indices)
                filtered = env.scene["cube_palm_contacts"].data.force_matrix_w[:, 0]
                peak_raw_pads = torch.maximum(peak_raw_pads, torch.linalg.vector_norm(raw, dim=-1))
                peak_filtered_pads = torch.maximum(peak_filtered_pads, torch.linalg.vector_norm(filtered, dim=-1))
                if step in (0, 1, args.contact_steps - 1):
                    debug_sample(f"{source}:{penetration_m}:step{step}")
                on = (load >= threshold) & bits[:, 0].bool() & (reward == 1.0)
                streak = torch.where(on, streak + 1, torch.zeros_like(streak))
                best_streak = torch.maximum(best_streak, streak)
                if bool((streak >= args.consecutive_on).all()):
                    accepted_gap = (source, penetration_m)
                    print("CENTRAL_PAD_REPEATED_ON", {"source": source, "penetration_m": penetration_m, "step": step,
                          "filtered_central_n": load.tolist(), "raw_palm_bit": bits[:, 0].tolist(),
                          "contact_reward": reward.tolist(), "consecutive_physics_steps": streak.tolist()}, flush=True)
                    break
            if accepted_gap is not None:
                break
        assert accepted_gap is not None, (
            f"No repeated physical central-pad ON at {threshold} N; "
            f"peaks={peak_central.tolist()}, best_streak={best_streak.tolist()}, "
            f"raw_pad_peaks={peak_raw_pads.tolist()}, filtered_pad_peaks={peak_filtered_pads.tolist()}"
        )

        # Explicit separation must clear both the filtered pad and reward.
        cube_force(gravity_compensation.expand(2, -1).clone())
        palm_position, hand_quaternion = safe_reset.palm_reference_pose_w()
        normal = quat_apply(hand_quaternion, hand_quaternion.new_tensor((0.0, 1.0, 0.0)).expand(2, -1))
        separated = palm_position + 0.30 * normal
        separated[:, 2] += 0.40
        write_pose(cube, separated, hand_quaternion)
        for _ in range(6):
            physics_step()
        assert bool((central_force() < threshold).all())
        assert not bool(cube_palm_contact_mask(env).any())
        assert not bool(palm_tactile_bits(env)[:, 0].any())
        assert not bool(palm_cube_contact_reward(env).any())
        print("CENTRAL_PAD_OFF", {"filtered_central_n": central_force().tolist(),
                                  "contact_reward": palm_cube_contact_reward(env).tolist()}, flush=True)

        # A deliberate table/forearm collision verifies real filtered pair
        # forces. This is a destructive test fixture inside this disposable
        # simulator session, and never becomes an environment reset state.
        forearm_ids, forearm_names = robot.find_bodies("forearm_link", preserve_order=True)
        assert tuple(forearm_names) == ("forearm_link",) and len(forearm_ids) == 1
        forearm_id = int(forearm_ids[0])
        forearm_position = robot.data.body_link_pos_w[:, forearm_id]
        forearm_quaternion = robot.data.body_link_quat_w[:, forearm_id]
        local_center = safe_reset.collision_bounds.centers[forearm_id].expand(2, -1)
        table_center = forearm_position + quat_apply(forearm_quaternion, local_center)
        # Keep the Cube away from the table and robot during this last stage.
        cube_away = env.scene.env_origins.clone()
        cube_away[:, 0] += 3.0
        cube_away[:, 2] += 2.5
        write_pose(cube, cube_away)
        write_pose(table, table_center)
        table_peak = torch.zeros(2, device=env.device)
        for _ in range(8):
            physics_step()
            history = env.scene["table_contacts"].data.force_matrix_w_history
            assert history.shape == (2, 2, 1, len(ROBOT_CONTACT_BODY_NAMES), 3), history.shape
            loads = torch.linalg.vector_norm(history, dim=-1)
            table_peak = torch.maximum(table_peak, loads.flatten(1).amax(-1))
            if bool(table_contact_failure(env).all()):
                break
        assert bool(table_contact_failure(env).all()), f"No actual robot-table contact: {table_peak.tolist()}"
        assert not bool(palm_cube_contact_reward(env).any())
        env.episode_length_buf[:] = env.max_episode_length
        assert not bool(contact_time_out(env).any()), "Timeout incorrectly won over table failure"
        env.termination_manager.compute()
        assert bool(env.termination_manager.terminated.all())
        assert not bool(env.termination_manager.time_outs.any())
        print("TABLE_FAILURE_PRECEDENCE", {"robot_pair_peak_n": table_peak.tolist(),
              "terminated": env.termination_manager.terminated.tolist(),
              "truncated": env.termination_manager.time_outs.tolist()}, flush=True)
        torch.testing.assert_close(env.command_manager.get_term("target_position").target_pos_w,
                                   initial_target, atol=0.0, rtol=0.0)
        print("CONTACT LIVE SENSOR PROBE PASS: controlled Cube loading; sensor/reward ON→repeated ON→OFF; "
              "Cube support excluded; real robot-table failure beats timeout. This does not test learned holding.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
