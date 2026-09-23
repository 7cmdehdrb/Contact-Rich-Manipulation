"""Run with env_isaaclab Python, or IsaacLab/isaaclab.sh -p this_file.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

# Also supports direct execution before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_tactile.model import build_model, repo_root
from inspire_tactile.controller import CYCLE_SECONDS, phase_at
from inspire_tactile.placement import DEFAULT_OBJECT_LOWERING, DEFAULT_OBJECT_OFFSET


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="UR5e + Inspire: rule-based grasp/release and 17 tactile bits")
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=0, help="0: repeat until GUI closes; positive: bounded run")
    parser.add_argument("--threshold", type=float, default=0.01, help="Force L2 norm threshold [N], inclusive >=")
    parser.add_argument("--print_hz", type=float, default=5.0)
    parser.add_argument("--ros2", action="store_true", help="Send to the separate ROS 2 bridge on localhost")
    parser.add_argument("--ros_port", type=int, default=9870)
    parser.add_argument("--ros_hz", type=float, default=30.0, help="ROS transport sampling frequency in simulation time")
    parser.add_argument("--log", type=Path, help="Optional JSONL output at print_hz; includes force vectors in N")
    parser.add_argument("--without_object", action="store_true", help="Negative control: no cylinder")
    parser.add_argument("--fix_object", action="store_true", help="Kinematic cylinder, for sensor-only checks")
    parser.add_argument("--object_offset", nargs=3, type=float, default=DEFAULT_OBJECT_OFFSET,
                        metavar=("X", "Y", "Z"), help="Fixed cylinder center in hand-base frame [m]")
    parser.add_argument("--object_lowering", type=float, default=DEFAULT_OBJECT_LOWERING,
                        help="Lower cylinder AND support along world Z [m]; default 0.02; 0 restores old height")
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--capture_dir", type=Path, help="Optional open/hold/released RGB snapshots")
    parser.add_argument("--check", action="store_true", help="Fail unless contact and subsequent release are measured")
    from inspire_tactile.projected_tactile import add_arguments
    add_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.num_envs < 1 or args.cycles < 0
            or not math.isfinite(args.print_hz) or args.print_hz <= 0
            or not math.isfinite(args.threshold) or args.threshold <= 0
            or not math.isfinite(args.ros_hz) or args.ros_hz <= 0
            or not 1 <= args.ros_port <= 65535
            or not math.isfinite(args.object_lowering) or args.object_lowering < 0
            or not math.isfinite(args.region_tolerance) or args.region_tolerance < 0
            or not all(math.isfinite(value) for value in args.object_offset)):
        parser.error("num_envs/print_hz/ros_hz/threshold must be positive; cycles/object_lowering non-negative; "
                     "all numeric options finite; ros_port within 1..65535")
    if args.check and args.cycles == 0:
        parser.error("--check requires --cycles >= 1")
    if args.capture_dir:
        args.enable_cameras = True
    return args


def run(args, app, urdf, model):
    import torch
    from pxr import PhysxSchema, Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from inspire_tactile.scene import make_scene
    from inspire_tactile.projected_tactile import make_reader

    dt = 1.0 / 240.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=4,
        physx=sim_utils.PhysxCfg(solve_articulation_contact_last=True),
    ))
    sim.set_camera_view(eye=(1.2, 1.0, 0.9), target=(0.35, 0.15, 0.43))
    scene = InteractiveScene(make_scene(
        urdf, model, num_envs=args.num_envs, object_offset=args.object_offset,
        object_lowering=args.object_lowering,
        without_object=args.without_object, fix_object=args.fix_object,
        force_conversion=args.force_conversion,
        tactile_mode=args.tactile_mode,
    ))
    camera = None
    if args.capture_dir:
        from isaaclab.sensors import Camera, CameraCfg
        camera = Camera(CameraCfg(
            prim_path="/World/InspectionCamera", height=600, width=800, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
        ))
        args.capture_dir.mkdir(parents=True, exist_ok=True)
    # The importer may instance collision subtrees even with make_instanceable=False.
    # Author millimetre-scale contact offsets on actual shapes, not instance proxies.
    for i in range(args.num_envs):
        root_prim = sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        for prim in list(Usd.PrimRange(root_prim)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        sim_utils.modify_collision_properties(
            str(root_prim.GetPath()),
            sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        )
    # Fail early if the importer merged a sensor or forgot contact reporting.
    for i in range(args.num_envs):
        for name in model["sensor_names"]:
            prim = sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{name}")
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Sensor body missing/merged: {name}; merge_fixed_joints must be False")
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                raise RuntimeError(f"Contact reporting not enabled on {name}")
            PhysxSchema.PhysxContactReportAPI(prim).CreateThresholdAttr().Set(0.0)
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[0.75, 0.75, 0.80]], device=sim.device),
            torch.tensor([[0.36, 0.15, 0.46]], device=sim.device),
        )
    robot = scene["robot"]
    reader = make_reader(scene, urdf, model, args, dt)
    target_open = robot.data.default_joint_pos.clone()
    target_closed = target_open.clone()
    for name, value in model["closed_targets"].items():
        target_closed[:, robot.joint_names.index(name)] = value
    print("TACTILE_CHANNELS " + json.dumps(dict(enumerate(reader.names))), flush=True)
    print(f"TACTILE_SOURCE {reader.force_source}; region tolerance={args.region_tolerance} m (observation only)", flush=True)
    print(f"TACTILE shape=({args.num_envs},17) dtype=torch.uint8; F_world [N]; threshold >= {args.threshold}", flush=True)
    print(f"MODE {'no-object control' if args.without_object else 'kinematic cylinder' if args.fix_object else 'dynamic cylinder'}", flush=True)

    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open("w")
        log.write(json.dumps({"type": "metadata", "channels": reader.names, "threshold_N": args.threshold,
                              "dt": dt, "cycle_seconds": CYCLE_SECONDS, "args": vars(args)}, default=str) + "\n")
    steps_per_cycle = round(CYCLE_SECONDS / dt)
    print_interval = max(1, round(1.0 / (args.print_hz * dt)))
    ros_interval = max(1, round(1.0 / (args.ros_hz * dt)))
    sender = None
    if args.ros2:
        from inspire_tactile.transport import TactileSender
        sender = TactileSender(args.ros_port)
        print(f"ROS transport: UDP 127.0.0.1:{args.ros_port}; start inspire_tactile_ros bridge + viewer", flush=True)
    peak = torch.zeros((args.num_envs, 17), device=sim.device)
    contact_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=sim.device)
    released_seen = torch.zeros_like(contact_seen)
    cycle_contact = torch.zeros_like(contact_seen)
    cycle_released = torch.zeros_like(contact_seen)
    cycle_results = []
    step, previous_phase = 0, None
    completed = 0
    try:
        while app.is_running() and (args.cycles == 0 or step < args.cycles * steps_per_cycle):
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.render()
                continue
            local_step = step % steps_per_cycle
            if local_step == 0:
                cycle_contact.zero_()
                cycle_released.zero_()
                # Only reset at cycle boundaries, never while closing/holding/releasing.
                robot.write_joint_state_to_sim(target_open, torch.zeros_like(target_open))
                if not args.without_object:
                    cylinder = scene["cylinder"]
                    state = cylinder.data.default_root_state.clone()
                    state[:, :3] += scene.env_origins
                    cylinder.write_root_pose_to_sim(state[:, :7])
                    cylinder.write_root_velocity_to_sim(state[:, 7:])
                scene.reset()
            phase, closure = phase_at(local_step * dt)
            robot.set_joint_position_target(target_open + closure * (target_closed - target_open))
            scene.write_data_to_sim()
            sim.step(render=not args.headless)
            scene.update(dt)
            forces, magnitude, bits = reader.read()
            if sender is not None and step % ros_interval == 0:
                sender.send(
                    time_s=(step + 1) * dt, phase=phase, names=reader.names, threshold=args.threshold,
                    positions=scene["tactile"].data.pos_w[:, reader.indices].cpu().tolist(),
                    forces=forces.cpu().tolist(), norms=magnitude.cpu().tolist(), bits=bits.cpu().tolist(),
                    force_source=reader.force_source, region_tolerance=args.region_tolerance,
                )
            if not torch.isfinite(robot.data.joint_pos).all() or (robot.data.joint_pos.abs() > 10).any():
                raise RuntimeError("Robot joint state diverged; stopping unstable simulation")
            peak = torch.maximum(peak, magnitude)
            any_contact = bits.bool().any(dim=-1)
            if phase in ("close", "hold"):
                contact_seen |= any_contact
                cycle_contact |= any_contact
            if phase == "released" and local_step * dt >= CYCLE_SECONDS - 0.5:
                released_seen |= contact_seen & ~any_contact
                cycle_released |= cycle_contact & ~any_contact
            if step % print_interval == 0 or phase != previous_phase:
                print(f"t={step*dt:6.2f}s phase={phase:8s} tactile={bits.cpu().tolist()} "
                      f"force_norm_N={[[round(x, 4) for x in row] for row in magnitude.cpu().tolist()]}",
                      flush=True)
                if log:
                    entry = {"type": "sample", "time_s": step * dt, "phase": phase,
                             "force_w_N": forces.cpu().tolist(), "norm_N": magnitude.cpu().tolist(),
                             "tactile": bits.cpu().tolist(), **reader.diagnostics(),
                             "joint_names": robot.joint_names,
                             "joint_pos": robot.data.joint_pos.cpu().tolist(),
                             "sensor_position_w": scene["tactile"].data.pos_w[:, reader.indices].cpu().tolist()}
                    if not args.without_object:
                        entry["object_position_w"] = scene["cylinder"].data.root_pos_w.cpu().tolist()
                        entry["object_state_w"] = scene["cylinder"].data.root_state_w.cpu().tolist()
                    log.write(json.dumps(entry) + "\n")
            previous_phase = phase
            if camera is not None and local_step in (240, 1320, 2520):
                import imageio.v3 as iio
                for _ in range(8):
                    sim.render()
                camera.update(dt, force_recompute=True)
                iio.imwrite(args.capture_dir / f"cycle_{step // steps_per_cycle:03d}_{phase}.png",
                            camera.data.output["rgb"][0].cpu().numpy())
            step += 1
            completed = step // steps_per_cycle
            if step % steps_per_cycle == 0:
                cycle_result = {"cycle": completed, "contact_seen": cycle_contact.cpu().tolist(),
                                "released_seen": cycle_released.cpu().tolist()}
                cycle_results.append(cycle_result)
                if log:
                    log.write(json.dumps({"type": "cycle_summary", **cycle_result}) + "\n")
                    log.flush()
        summary = {"type": "summary", "completed_cycles": completed,
                   "contact_seen": contact_seen.cpu().tolist(), "released_seen": released_seen.cpu().tolist(),
                   "peak_norm_N": peak.cpu().tolist(), "cycles": cycle_results}
        print("SUMMARY " + json.dumps(summary), flush=True)
        if log:
            log.write(json.dumps(summary) + "\n")
        if args.check:
            if completed < args.cycles:
                raise RuntimeError("Check interrupted before requested cycles completed")
            if args.without_object:
                if (peak >= args.threshold).any():
                    raise RuntimeError("Negative control has tactile contact: inspect pedestal/self collisions")
            elif not all(all(result["contact_seen"]) and all(result["released_seen"]) for result in cycle_results):
                raise RuntimeError("Expected measured contact and release in every environment and every cycle")
    finally:
        if sender is not None:
            sender.close()
        if log:
            log.close()


def main():
    args = arguments()
    root = (args.repo_root or repo_root()).resolve()
    urdf, model = build_model(root, root / "src/inspire_tactile/generated")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    exit_code = 0
    try:
        run(args, launcher.app, urdf, model)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Artifacts are synchronously written. Avoid Isaac Sim 5.1 GPU stage-close
        # deadlocks by using its documented standalone fast-shutdown path.
        # Fast shutdown exits the process: propagate failures to Kit BEFORE close.
        import omni.kit.app
        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
