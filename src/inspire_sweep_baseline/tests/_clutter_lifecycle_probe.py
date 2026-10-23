"""Headless Isaac/PhysX contract probe for Contact Sweep Clutter.

This is an intentionally manipulated-state probe, not a learned policy or a
natural-contact success evaluation. Run explicitly with the Isaac environment.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from inspire_sweep_baseline.assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, TACTILE_REGIONS, parent_sensor_name
from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter import all_object_tilt_cost
from inspire_sweep_baseline.mdp.reward_functions.goal_rewards import object_tilt_cost


def read_cylinder_properties(env):
    objects = env.scene["object_collection"]
    view = objects.root_physx_view
    return {
        "mass": objects.reshape_view_to_data(view.get_masses().clone()),
        "inertia": objects.reshape_view_to_data(view.get_inertias().clone()),
        "material": view.get_material_properties().clone().reshape(
            objects.num_objects, env.num_envs, view.max_shapes, 3,
        ).transpose(0, 1),
    }


def check_cylinder_equality(properties):
    for name, values in properties.items():
        expected = values[:, :1].expand_as(values)
        assert torch.equal(values, expected), f"Objects disagree on actual PhysX {name}"


task = "Isaac-Inspire-Contact-Sweep-Clutter-v0"
try:
    with torch.inference_mode():
        assert args.num_envs >= 4, "The partial-reset probe needs at least four environments"
        cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        cfg.seed = 42
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            observations, _ = env.reset(seed=42)
            robot, objects = env.scene["robot"], env.scene["object_collection"]
            seeded_positions = objects.data.object_pos_w.clone()
            seeded_snapshot = observations["obstacles"].clone()
            observations, _ = env.reset(seed=42)
            assert torch.equal(objects.data.object_pos_w, seeded_positions)
            assert torch.equal(observations["obstacles"], seeded_snapshot)
            assert objects.num_objects == 4
            assert objects.object_names == ["mass_cylinder", *[f"obstacle_{i}" for i in range(3)]]
            assert observations["policy"].shape == (env.num_envs, 57)
            assert observations["obstacles"].shape == (env.num_envs, 40)
            snapshot = observations["obstacles"].clone()
            slots = snapshot.reshape(env.num_envs, 10, 4)
            assert torch.equal(slots[:, :3, 0], torch.ones_like(slots[:, :3, 0]))
            assert slots[:, 3:].count_nonzero() == 0
            root_positions, _ = subtract_frame_transforms(
                robot.data.root_pos_w[:, None].expand(-1, 3, -1),
                robot.data.root_quat_w[:, None].expand(-1, 3, -1),
                objects.data.object_pos_w[:, 1:],
            )
            noise = slots[:, :3, 1:] - root_positions
            assert noise.abs().max() <= 0.010001
            goal = env.command_manager.get_command("target_goal_pos")[:, :3]
            assert torch.allclose(goal, objects.data.object_pos_w[:, 0] + env.sweep_dir, atol=1e-6)
            properties = read_cylinder_properties(env)
            check_cylinder_equality(properties)

            arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
            hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
            velocity_limits = robot.root_physx_view.get_dof_max_velocities()
            effort_limits = robot.root_physx_view.get_dof_max_forces()
            expected_arm = torch.tensor([2., 2., 2., 4., 4., 4.], device=velocity_limits.device)
            assert torch.equal(velocity_limits[:, arm_ids], expected_arm.expand(env.num_envs, -1))
            assert torch.equal(velocity_limits[:, hand_ids], torch.ones_like(velocity_limits[:, hand_ids]))
            expected_effort = torch.tensor([150., 150., 150., 28., 28., 28.], device=effort_limits.device)
            assert torch.equal(effort_limits[:, arm_ids], expected_effort.expand(env.num_envs, -1))
            assert torch.allclose(effort_limits[:, hand_ids], torch.full_like(effort_limits[:, hand_ids], 0.2))

            filters = {}
            for parent in TACTILE_PARENT_BODY_NAMES:
                sensor = env.scene[parent_sensor_name(parent)]
                raw = sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
                assert raw[4].shape == (env.num_envs, 4)
                filters[parent] = raw[4].shape[1]

            # Run real steps; rows that reset naturally get a new valid snapshot.
            actions = torch.zeros((env.num_envs, 8), device=env.device)
            actions[:, 6:] = 1.0
            stable_transitions = 0
            natural_resets = 0
            for _ in range(5):
                before = env._initial_obstacle_observation.clone()
                observations, _, terminated, timeout, _ = env.step(actions)
                reset = terminated | timeout
                assert torch.equal(observations["obstacles"][~reset], before[~reset])
                stable_transitions += int((~reset).sum())
                natural_resets += int(reset.sum())
            assert stable_transitions > 0

            # Moving obstacle truth does not update the initial-only observation.
            state = objects.data.object_link_state_w.clone()
            state[:, 1, 0] += 0.02
            objects.write_object_link_state_to_sim(state)
            cached = env._initial_obstacle_observation.clone()
            assert torch.equal(env.observation_manager.compute()["obstacles"], cached)

            selected = torch.tensor([1, 3], device=env.device)
            remaining = torch.tensor([i for i in range(env.num_envs) if i not in (1, 3)], device=env.device)
            old_properties = read_cylinder_properties(env)
            old_states = objects.data.object_link_state_w.clone()
            env.reset(env_ids=selected)
            assert torch.equal(env._initial_obstacle_observation[remaining], cached[remaining])
            assert not torch.equal(env._initial_obstacle_observation[selected], cached[selected])
            assert torch.equal(objects.data.object_link_state_w[remaining], old_states[remaining])
            new_properties = read_cylinder_properties(env)
            check_cylinder_equality(new_properties)
            for name in old_properties:
                ids = remaining.to(old_properties[name].device)
                assert torch.equal(new_properties[name][ids], old_properties[name][ids]), name
            assert not torch.equal(
                new_properties["mass"][selected.cpu()], old_properties["mass"][selected.cpu()],
            )

            # Exercise the registered manager terms after tipping an obstacle
            # and dropping a different obstacle, with the target left upright.
            env.reset()
            baseline_drop = env.termination_manager.get_term_cfg("object_drop")
            assert not baseline_drop.func(env, **baseline_drop.params).any()
            state = objects.data.object_link_state_w.clone()
            roll = torch.tensor([1.1], device=env.device)
            state[0, 1, 3:7] = quat_from_euler_xyz(roll, torch.zeros_like(roll), torch.zeros_like(roll))[0]
            state[1, 2, 2] = env.scene.env_origins[1, 2] + 1.0
            objects.write_object_link_state_to_sim(state)
            env.termination_manager.compute()
            drop_result = env.termination_manager.get_term("object_drop")
            assert drop_result[0] and drop_result[1] and not drop_result[2:].any()
            expected_tilt = torch.zeros(env.num_envs, device=env.device)
            for object_id in range(objects.num_objects):
                env.target_id[:] = object_id
                expected_tilt += object_tilt_cost(env)
            env.target_id.zero_()
            assert torch.allclose(all_object_tilt_cost(env), expected_tilt, atol=1e-6)
            env.reward_manager.compute(env.step_dt)
            tilt_index = env.reward_manager.active_terms.index("object_tilt")
            assert torch.allclose(
                env.reward_manager._step_reward[:, tilt_index], -2.0 * expected_tilt, atol=1e-6,
            )

            # A controlled physical contact establishes that an obstacle enters
            # the new parent-filter column and activates the existing palm bit.
            env.reset()
            region = TACTILE_REGIONS[0]
            assert region["name"] == "inspire_palm_force_sensor"
            pad_id = robot.body_names.index(region["name"])
            parent_sensor = env.scene[parent_sensor_name(region["parent_name"])]
            tactile_index = env.observation_manager.active_terms["policy"].index("tactile")
            tactile_term = env.observation_manager._group_obs_term_cfgs["policy"][tactile_index]
            fixed_joint_pos = robot.data.joint_pos.clone()
            contact_seen = palm_seen = False
            contact_columns = torch.zeros(objects.num_objects, dtype=torch.long, device=env.device)
            offsets = [(x, y) for x, y in (
                (0., 0.), (0.02, 0.), (-0.02, 0.), (0.04, 0.), (-0.04, 0.),
                (0., 0.02), (0., -0.02), (0.02, 0.02), (-0.02, -0.02),
            )]
            contact_diagnostics = []
            for step, (offset_x, offset_y) in enumerate(offsets):
                robot.write_joint_state_to_sim(fixed_joint_pos, torch.zeros_like(robot.data.joint_vel))
                # Hold unrelated cylinders away so no target contact can explain
                # the signal. The manipulated cylinder still has its real mass.
                contact_states = objects.data.object_link_state_w.clone()
                contact_states[..., :3] = env.scene.env_origins[:, None] + torch.tensor(
                    [0., 2., 4.], device=env.device,
                )
                contact_states[..., 0] += torch.arange(objects.num_objects, device=env.device)[None] * 0.15
                contact_states[..., 3:7] = contact_states.new_tensor([1., 0., 0., 0.])
                contact_states[..., 7:] = 0.
                pad_pos = robot.data.body_link_pos_w[0, pad_id]
                pad_quat = robot.data.body_link_quat_w[0, pad_id]
                local_point = torch.tensor(region["verification"]["parent_surface_point_local_m"], device=env.device)
                local_normal = torch.tensor(
                    region["verification"]["parent_surface_outward_normal_local"], device=env.device,
                )
                world_point = pad_pos + quat_apply(pad_quat, local_point)
                world_normal = quat_apply(pad_quat, local_normal)
                offset = quat_apply(pad_quat, local_point.new_tensor([offset_x, offset_y, 0.]))
                contact_states[0, 1, :3] = world_point + offset - 0.002 * world_normal
                # Align the cylinder's end face with the calibrated parent
                # surface, whose normal differs slightly from the pad's +Z.
                local_up = world_normal.new_tensor([0., 0., 1.])
                cylinder_quat = torch.cat((
                    (1. + world_normal[2]).reshape(1),
                    torch.linalg.cross(local_up, world_normal),
                ))
                contact_states[0, 1, 3:7] = cylinder_quat / torch.linalg.vector_norm(cylinder_quat)
                objects.write_object_link_state_to_sim(contact_states)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                env.sensor_data_fresh[:] = True
                raw = parent_sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
                contact_columns += raw[4][0].to(torch.long)
                bits = tactile_term.func(env, **tactile_term.params)
                contact_seen |= bool(raw[4][0, 1] > 0)
                palm_seen |= bool(bits[0, 0] > 0)
                start, count = int(raw[5][0, 1]), int(raw[4][0, 1])
                local_contacts = quat_apply_inverse(
                    pad_quat.expand(count, -1), raw[1][start:start + count] - pad_pos,
                )
                contact_diagnostics.append({
                    "offset": [offset_x, offset_y], "points_pad_frame": local_contacts.tolist(),
                    "active_bits": bits[0].nonzero().flatten().tolist(),
                })
                assert raw[4][0, 0] == 0, "Unexpected target contact contaminated the obstacle probe"
                if contact_seen and palm_seen:
                    break
            assert contact_seen, f"No obstacle_0 parent contact observed: {contact_columns.tolist()}"
            assert palm_seen, f"Obstacle palm probe: {contact_diagnostics}"

            report = {
                "task": task,
                "num_envs": env.num_envs,
                "physical_objects": objects.num_objects,
                "observation_shapes": {"policy": [env.num_envs, 57], "obstacles": [env.num_envs, 40]},
                "real_obstacle_slots": 3,
                "zero_dummy_slots": 7,
                "initial_observation_max_noise_m": float(noise.abs().max()),
                "identical_actual_mass_inertia_material": True,
                "arm_velocity_limits_rad_s": velocity_limits[0, arm_ids].tolist(),
                "hand_velocity_limits_rad_s": velocity_limits[0, hand_ids].tolist(),
                "stable_step_transitions": stable_transitions,
                "natural_reset_count": natural_resets,
                "partial_reset_indices": [1, 3],
                "partial_reset_preserved_other_rows": True,
                "goal_equals_initial_target_plus_direction": True,
                "repeated_explicit_reset_seed_reproducible": True,
                "obstacle_drop_and_tilt_manager_checks": True,
                "parent_filter_columns": filters,
                "obstacle_contact_counts_per_column": contact_columns.tolist(),
                "obstacle_palm_tactile_detected": palm_seen,
                "controlled_contact_steps": step + 1,
                "manipulated_states": True,
                "passed": True,
            }
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("CLUTTER_LIFECYCLE " + json.dumps(report), flush=True)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            raise
        finally:
            env.close()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.app.close()
