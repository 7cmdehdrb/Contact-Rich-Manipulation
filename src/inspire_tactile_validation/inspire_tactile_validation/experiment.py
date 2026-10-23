"""Vectorized probe sequence and raw evidence collection (requires Kit)."""

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat
from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES, ROBOT_USD_PATH
from inspire_sweep_baseline.mdp.observation_functions.tactile_observations import ProjectedTactileContactBits

from .scene import make_scene
from .targets import build_targets, load_regions, DEFAULT_REGION_PATH
from .reporting import summarize_trial, write_results
from .stepping import advance_physics


def run(args, app):
    targets = [t for t in build_targets(args.corner_inset) if args.channels is None or t["channel"] in args.channels]
    jobs = [dict(t, repeat=repeat, trial_id=index + repeat * len(targets))
            for repeat in range(args.repeats) for index, t in enumerate(targets)]
    n = min(args.num_envs, len(jobs))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=2, gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.2, friction_correlation_distance=0.00625,
                                enable_external_forces_every_iteration=args.external_forces_every_iteration,
                                solver_type=1 if args.solver == "TGS" else 0),
    ))
    sim.set_camera_view(eye=(0.6, 0.6, 1.0), target=(0.0, 0.0, 0.35))
    scene = InteractiveScene(make_scene(n, spacing=0.7, probe_radius=args.probe_radius, probe_length=args.probe_length, pad_layout=args.pad_layout, pad_clearance=args.pad_clearance))
    sim.reset()
    hand, probe = scene["robot"], scene["probe"]
    root_pose = hand.data.default_root_state[:, :7].clone()
    root_pose[:, :3] += scene.env_origins
    hand.write_root_pose_to_sim(root_pose)
    hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(), hand.data.default_joint_vel.clone())
    hand.set_joint_position_target(hand.data.default_joint_pos)
    adapter = SimpleNamespace(scene=scene, device=sim.device, num_envs=n, physics_dt=dt,
                              sensor_data_fresh=torch.zeros(n, dtype=torch.bool, device=sim.device))
    reader = ProjectedTactileContactBits(SimpleNamespace(params={"sensor_name": "hand_tactile"}), adapter)
    pad_ids = torch.tensor([hand.body_names.index(name) for name in PALM_SENSOR_BODY_NAMES], device=sim.device)
    sensor_ids = torch.tensor([scene["hand_tactile"].body_names.index(name) for name in PALM_SENSOR_BODY_NAMES], device=sim.device)
    body_names = list(HAND_CONTACT_BODY_NAMES)
    regions = load_regions()
    edge_count = max(len(r["footprint_halfspaces"]) for r in regions)
    region_planes = torch.zeros((17, edge_count, 3), device=sim.device)
    region_planes[..., 2] = -1.0
    for index, region in enumerate(regions):
        values = region_planes.new_tensor(region["footprint_halfspaces"])
        region_planes[index, :len(values)] = values
    meta = {
        "schema_version": 2, "pad_layout": args.pad_layout, "gravity_world": [0.0, 0.0, 0.0],
        "render_independent_physics": True, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "channels": list(PALM_SENSOR_BODY_NAMES), "targets": targets, "physics_dt": dt,
        "reader": "inspire_sweep_baseline.mdp.observation_functions.tactile_observations.ProjectedTactileContactBits",
        "region_sha256": hashlib.sha256(DEFAULT_REGION_PATH.read_bytes()).hexdigest(),
        "test_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "usd_sha256": {str(p.relative_to(ROBOT_USD_PATH.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROBOT_USD_PATH.parent.rglob("*.usd")},
        "notes": ["Original meshes and reader; test-only pad placement is recorded explicitly. Probe replaces cylinder filter.",
                  "Dynamic capsule with capped Cartesian force servo and inertia-scaled orientation servo.",
                  "Force evidence is probe contact force, not a commanded-force or aim-position assumption.",
                  "This validates simulated contact detection, not real tactile pressure calibration."],
    }
    try:
        meta["workspace_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        meta["workspace_diff"] = subprocess.check_output(["git", "diff", "--stat"], text=True)
    except (OSError, subprocess.CalledProcessError):
        meta["workspace_commit"] = None
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))

    physics_ticks = 0
    def step():
        nonlocal physics_ticks
        physics_ticks += 1
        advance_physics(sim, scene, dt, render=sim.has_gui() and physics_ticks % 2 == 0)

    for _ in range(math.ceil(args.warmup_seconds / dt)):
        if not app.is_running():
            raise RuntimeError("Simulation closed during warmup")
        step()
    # Confirm the fixture's palm is facing upward, independently of mesh naming.
    palm_q = hand.data.body_link_quat_w[:, pad_ids[0]]
    palm_normal = quat_apply(palm_q, palm_q.new_tensor([0., 0., 1.]).expand(n, 3))
    if not bool((palm_normal[:, 2] > 0.99).all()):
        raise RuntimeError(f"Palm-up fixture failed: {palm_normal.cpu().tolist()}")
    if hand.num_joints != 12:
        raise RuntimeError(f"Hand-only fixture must have 12 joints, got {hand.num_joints}")
    root_relative = hand.data.root_pos_w - scene.env_origins
    if not bool(torch.allclose(root_relative, root_relative[:1].expand_as(root_relative), atol=1e-4)):
        raise RuntimeError("Cloned hands do not share the same local root pose")
    meta["hand_roots_relative_to_env"] = root_relative.cpu().tolist()
    meta["pad_lifts_m"] = {
        name: sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}").GetAttribute("validation:padLiftM").Get() or 0.0
        for name in PALM_SENSOR_BODY_NAMES
    }
    meta["hand_joint_names"] = hand.joint_names
    meta["hand_body_names"] = hand.body_names
    meta["palm_normal_world"] = palm_normal.cpu().tolist()
    phases = [(name, max(1, math.ceil(getattr(args, name + "_seconds") / dt)))
              for name in ("baseline", "approach", "hold", "retract", "recovery")]
    trials = []
    completed = True
    with (output / "samples.csv").open("w", newline="") as stream:
        fields = ["trial_id", "env_id", "phase", "time_s", "probe_force_n", "probe_target_contact", "commanded_force_n", "target_distance_m"]
        fields += ["bit_" + name for name in PALM_SENSOR_BODY_NAMES]
        fields += ["direct_n_" + name for name in PALM_SENSOR_BODY_NAMES]
        fields += ["target_probe_force_n", "contact_local_x_m", "contact_local_y_m", "contact_local_z_m", "contact_aim_xy_error_m", "hand_joint_error_rad", "probe_speed_m_s", "force_balance_n"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for begin in range(0, len(jobs), n):
            batch = jobs[begin:begin + n]
            count = len(batch)
            padded = batch + [batch[-1]] * (n - count)
            channels = torch.tensor([t["channel"] for t in padded], device=sim.device)
            rows = torch.arange(n, device=sim.device)
            adapter.sensor_data_fresh[:] = False
            # Restore fully open, motionless hand before every independent trial.
            hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(), torch.zeros_like(hand.data.default_joint_vel))
            hand.set_joint_position_target(hand.data.default_joint_pos)
            # Clear the previous probe before settling the new open hand.
            parked = probe.data.default_root_state[:, :7].clone()
            parked[:, :3] += scene.env_origins
            probe.write_root_pose_to_sim(parked)
            probe.write_root_velocity_to_sim(torch.zeros((n, 6), device=sim.device))
            probe.permanent_wrench_composer.set_forces_and_torques(forces=torch.zeros((n, 1, 3), device=sim.device), torques=torch.zeros((n, 1, 3), device=sim.device), is_global=True)
            for _ in range(math.ceil(args.warmup_seconds / dt)):
                step()
            pad_pos = hand.data.body_link_pos_w[rows, pad_ids[channels]].clone()
            pad_quat = hand.data.body_link_quat_w[rows, pad_ids[channels]].clone()
            local = torch.tensor([t["local_point"] for t in padded], device=sim.device)
            normal = quat_apply(pad_quat, torch.tensor([t["normal"] for t in padded], device=sim.device))
            aim = pad_pos + quat_apply(pad_quat, local)
            # Begin beyond the complete projection extrusion, including buried pads.
            outer = torch.tensor([max(0.0, regions[t["channel"]]["outward_z_max_m"] - t["local_point"][2]) for t in padded], device=sim.device)
            start_tip = aim + normal * (outer + args.clearance)[:, None]
            end_tip = aim - args.indentation * normal
            half_length = args.probe_length / 2.0 + args.probe_radius
            center_start = start_tip + normal * half_length
            probe.write_root_pose_to_sim(torch.cat((center_start, pad_quat), dim=-1))
            probe.write_root_velocity_to_sim(torch.zeros((n, 6), device=sim.device))
            scene.reset()
            samples = [[] for _ in batch]
            wanted_body_ids = torch.tensor([[body_names.index(t["name"]), body_names.index(t["parent"])] for t in padded], device=sim.device)
            target_planes = region_planes[channels]
            elapsed = 0
            for phase, steps in phases:
                for tick in range(steps):
                    if not app.is_running():
                        completed = False
                        break
                    wall_start = time.perf_counter()
                    alpha = (tick + 1) / steps
                    if phase == "approach":
                        tip = torch.lerp(start_tip, end_tip, alpha)
                    elif phase == "hold":
                        tip = end_tip
                    elif phase == "retract":
                        tip = torch.lerp(end_tip, start_tip, alpha)
                    else:
                        tip = start_tip
                    desired_center = tip + half_length * normal
                    mass = 0.05
                    force = args.stiffness * (desired_center - probe.data.root_pos_w) - 2.0 * math.sqrt(args.stiffness * mass) * probe.data.root_lin_vel_w
                    force *= (args.max_force / force.norm(dim=-1).clamp_min(1e-9)).clamp_max(1.0)[:, None]
                    orientation_error = axis_angle_from_quat(quat_mul(pad_quat, quat_conjugate(probe.data.root_quat_w)))
                    inertia = force.new_tensor([mass * (args.probe_length ** 2 / 12.0 + args.probe_radius ** 2 / 4.0)] * 2 + [0.5 * mass * args.probe_radius ** 2])
                    angular_command = 30.0 ** 2 * orientation_error - 60.0 * probe.data.root_ang_vel_w
                    torque = quat_apply(probe.data.root_quat_w, inertia * quat_apply_inverse(probe.data.root_quat_w, angular_command))
                    probe.permanent_wrench_composer.set_forces_and_torques(forces=force[:, None], torques=torque[:, None], is_global=True)
                    previous_velocity = probe.data.root_lin_vel_w.clone()
                    step()
                    balance = (mass * (probe.data.root_lin_vel_w - previous_velocity) / dt - force).norm(dim=-1)
                    adapter.sensor_data_fresh[:] = True
                    bits = reader(adapter, threshold_n=args.threshold)
                    direct = scene["hand_tactile"].data.net_forces_w[:, sensor_ids].norm(dim=-1)
                    contact = scene["probe_contact"].data
                    probe_load = contact.net_forces_w[:, 0].norm(dim=-1)
                    matrices = contact.force_matrix_w[:, 0]
                    selected_forces = matrices[rows[:, None], wanted_body_ids]
                    intended_load = selected_forces.sum(dim=1).norm(dim=-1)
                    selected_points = torch.nan_to_num(contact.contact_pos_w[:, 0][rows[:, None], wanted_body_ids], nan=0.0)
                    weights = selected_forces.norm(dim=-1)
                    centroid = (selected_points * weights[..., None]).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)[:, None]
                    live_pos = hand.data.body_link_pos_w[rows, pad_ids[channels]]
                    live_quat = hand.data.body_link_quat_w[rows, pad_ids[channels]]
                    local_contact = quat_apply_inverse(live_quat, centroid - live_pos)
                    aim_error = (local_contact[:, :2] - local[:, :2]).norm(dim=-1)
                    in_region = ((target_planes[..., :2] * local_contact[:, None, :2]).sum(dim=-1) + target_planes[..., 2] <= 0.0005).all(dim=-1)
                    # Verify independent load on the correct parent/pad AND XY
                    # footprint. Do not reuse the reader's Z/normal gates.
                    target_contact = (intended_load >= args.threshold) & in_region
                    joint_error = (hand.data.joint_pos - hand.data.default_joint_pos).abs().amax(dim=-1)
                    actual_normal = quat_apply(probe.data.root_quat_w, normal.new_tensor([0.0, 0.0, 1.0]).expand(n, 3))
                    actual_tip = probe.data.root_pos_w - actual_normal * half_length
                    packed = torch.cat((bits, direct, probe_load[:, None], target_contact[:, None], force.norm(dim=-1)[:, None], (actual_tip - aim).norm(dim=-1)[:, None], intended_load[:, None], local_contact, aim_error[:, None], joint_error[:, None], probe.data.root_lin_vel_w.norm(dim=-1)[:, None], balance[:, None]), dim=-1).cpu().tolist()
                    for i, values in enumerate(packed[:count]):
                        record = dict(phase=phase, time_s=elapsed * dt,
                                      bits=values[:17], direct_forces_n=values[17:34],
                                      probe_force_n=values[34], probe_contact=values[34] > 1e-6,
                                      probe_target_contact=bool(values[35]),
                                      probe_speed_m_s=values[44], force_balance_n=values[45])
                        samples[i].append(record)
                        row = dict(trial_id=batch[i]["trial_id"], env_id=i, phase=phase, time_s=elapsed * dt,
                                   probe_force_n=values[34], probe_target_contact=bool(values[35]), commanded_force_n=values[36], target_distance_m=values[37])
                        row.update(zip(fields[8:25], values[:17]))
                        row.update(zip(fields[25:42], values[17:34]))
                        row.update(zip(fields[42:], values[38:46]))
                        writer.writerow(row)
                    elapsed += 1
                    if args.real_time:
                        time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))
                if not completed:
                    break
            if not completed:
                break
            for t, records in zip(batch, samples):
                trials.append(dict(t, **summarize_trial(records, t["channel"], args.threshold)))
            stream.flush()
            print(f"[tactile] completed {len(trials)}/{len(jobs)} trials", flush=True)
    meta["collection_complete"] = completed
    meta["expected_trials"] = len(jobs)
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    summary = write_results(output, trials, meta)
    print(f"[tactile] results: {output}; {summary['status_counts']}", flush=True)
    # Disable the standalone pause loop before stopping for batch shutdown.
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()
    return summary["status"] == "completed" and summary["criteria_status"] == "pass"
