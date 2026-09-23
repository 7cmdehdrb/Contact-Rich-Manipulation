"""Individual hand joint sliders in Isaac Sim; no automatic grasp/release cycle."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspire_tactile.manual_control import JointControls
from inspire_tactile.model import build_model, repo_root
from inspire_tactile.placement import DEFAULT_OBJECT_LOWERING, DEFAULT_OBJECT_OFFSET


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path)
    parser.add_argument("--duration", type=float, default=0, help="Simulation seconds; 0: until GUI closes")
    parser.add_argument("--speed", type=float, default=0.6, help="Joint target slew rate [rad/s], capped by URDF velocity")
    parser.add_argument("--linked", action="store_true", help="Start with original mimic coupling (6 motor targets)")
    parser.add_argument("--joint", action="append", default=[], metavar="NAME=DEGREES", help="Initial target, repeatable")
    parser.add_argument("--without_object", action="store_true")
    parser.add_argument("--fix_object", action="store_true")
    parser.add_argument("--object_offset", nargs=3, type=float, default=DEFAULT_OBJECT_OFFSET)
    parser.add_argument("--object_lowering", type=float, default=DEFAULT_OBJECT_LOWERING)
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--print_hz", type=float, default=5)
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--ros_port", type=int, default=9870)
    parser.add_argument("--ros_hz", type=float, default=30)
    parser.add_argument("--log", type=Path)
    from inspire_tactile.projected_tactile import add_arguments
    add_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.speed, args.threshold, args.print_hz, args.ros_hz)):
        parser.error("speed/threshold/print_hz/ros_hz must be positive and finite")
    if any(not math.isfinite(v) or v < 0 for v in (args.duration, args.object_lowering, args.region_tolerance)):
        parser.error("duration/object_lowering/region_tolerance must be non-negative and finite")
    if not all(math.isfinite(v) for v in args.object_offset) or not 1 <= args.ros_port <= 65535:
        parser.error("object_offset must be finite; ros_port must be within 1..65535")
    if args.headless and args.duration == 0:
        parser.error("Headless mode needs --duration > 0 (no interactive sliders)")
    return args


def run(args, app, urdf, model, controls):
    import torch
    from pxr import PhysxSchema, Usd, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from inspire_tactile.scene import make_scene
    from inspire_tactile.projected_tactile import make_reader
    from inspire_tactile.transport import TactileSender

    dt = 1 / 240
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, render_interval=4,
        physx=sim_utils.PhysxCfg(solve_articulation_contact_last=True)))
    sim.set_camera_view(eye=(1.2, 1.0, 0.9), target=(0.35, 0.15, 0.43))
    scene = InteractiveScene(make_scene(
        urdf, model, num_envs=1, object_offset=args.object_offset, object_lowering=args.object_lowering,
        without_object=args.without_object, fix_object=args.fix_object, force_conversion=args.force_conversion,
        tactile_mode=args.tactile_mode))
    root = sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
    for prim in list(Usd.PrimRange(root)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    sim_utils.modify_collision_properties(str(root.GetPath()),
        sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0))
    for name in model["sensor_names"]:
        prim = sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise RuntimeError(f"Sensor missing or contact reporting disabled: {name}")
        PhysxSchema.PhysxContactReportAPI(prim).CreateThresholdAttr().Set(0.0)
    sim.reset()
    robot = scene["robot"]
    reader = make_reader(scene, urdf, model, args, dt)
    indices = [robot.joint_names.index(n) for n in controls.names]
    target = robot.data.default_joint_pos.clone()  # arm never receives slider commands
    # sim.reset() creates physics handles, but does not place imported joints at
    # the configured default pose. Initialize explicitly, as play.py does.
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.set_joint_position_target(target)
    scene.reset()
    panel = sender = log = None
    step, resets = 0, 0
    try:
        if not args.headless:
            from inspire_tactile.manual_ui import JointPanel
            panel = JointPanel(controls)
        if args.ros2:
            sender = TactileSender(args.ros_port)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("w")
            log.write(json.dumps({"type": "metadata", "mode": "manual", "names": controls.names,
                                  "specs": controls.specs, "args": vars(args)}, default=str) + "\n")
        print("MANUAL_JOINTS " + json.dumps(controls.specs), flush=True)
        print("TACTILE_CHANNELS " + json.dumps(dict(enumerate(reader.names))), flush=True)
        print(f"TACTILE_SOURCE {reader.force_source}; region tolerance={args.region_tolerance} m (observation only)", flush=True)
        print("MANUAL: 12 independent joints; enable Link mimic for the original 6-motor arrangement.", flush=True)
        interval = max(1, round(1 / (args.print_hz * dt)))
        ros_interval = max(1, round(1 / (args.ros_hz * dt)))
        while app.is_running() and (args.duration == 0 or step * dt < args.duration):
            if sim.is_stopped() or (panel is not None and not panel.window.visible):
                break
            if not sim.is_playing():
                sim.render()
                continue
            if controls.reset_requested:
                controls.reset()
                robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(target))
                if not args.without_object:
                    state = scene["cylinder"].data.default_root_state.clone()
                    state[:, :3] += scene.env_origins
                    scene["cylinder"].write_root_pose_to_sim(state[:, :7])
                    scene["cylinder"].write_root_velocity_to_sim(state[:, 7:])
                scene.reset()
                resets += 1
                if panel:
                    panel.sync()
            command = controls.step(dt)
            target[:, indices] = torch.tensor([command[n] for n in controls.names], device=sim.device)
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step(render=not args.headless)
            scene.update(dt)
            step += 1
            if not torch.isfinite(robot.data.joint_pos).all() or (robot.data.joint_pos.abs() > 10).any():
                raise RuntimeError("Robot joint state diverged")
            forces, norms, bits = reader.read()
            if sender and step % ros_interval == 0:
                sender.send(time_s=step * dt, phase="manual", names=reader.names, threshold=args.threshold,
                            positions=scene["tactile"].data.pos_w[:, reader.indices].cpu().tolist(),
                            forces=forces.cpu().tolist(), norms=norms.cpu().tolist(), bits=bits.cpu().tolist(),
                            force_source=reader.force_source, region_tolerance=args.region_tolerance)
            if step % interval == 0:
                actual = dict(zip(controls.names, robot.data.joint_pos[0, indices].cpu().tolist()))
                binary = bits[0].cpu().tolist()
                if panel:
                    panel.refresh(actual, binary, step * dt)
                print(f"t={step*dt:.2f}s manual tactile={binary}", flush=True)
                if log:
                    entry = {"type": "sample", "time_s": step * dt, "phase": "manual", "linked": controls.linked,
                             "desired_rad": controls.desired, "command_rad": command, "actual_rad": actual,
                             "arm_actual_rad": {n: float(robot.data.joint_pos[0, robot.joint_names.index(n)])
                                                for n in model["arm_pose"]},
                             "tactile": binary, "norm_N": norms[0].cpu().tolist(), "resets": resets,
                             "force_w_N": forces.cpu().tolist(), **reader.diagnostics()}
                    if not args.without_object:
                        entry["object_state_w"] = scene["cylinder"].data.root_state_w.cpu().tolist()
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
        print(f"MANUAL_DONE steps={step} resets={resets}", flush=True)
    finally:
        if panel:
            panel.destroy()
        if sender:
            sender.close()
        if log:
            log.close()


def main():
    args = arguments()
    root = (args.repo_root or repo_root()).resolve()
    urdf, model = build_model(root, root / "src/inspire_tactile/generated")
    controls = JointControls(model, args.speed, args.linked)
    for assignment in args.joint:
        name, value = assignment.split("=", 1)
        if not name.startswith("inspire_"):
            name = "inspire_" + name
        controls.set_target(name, math.radians(float(value)))
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    exit_code = 0
    try:
        run(args, launcher.app, urdf, model, controls)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        import omni.kit.app
        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
