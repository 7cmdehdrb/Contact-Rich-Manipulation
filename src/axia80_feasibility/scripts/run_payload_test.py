"""Run the vectorized UR5e--Axia80--tool static payload experiment.

The default mass sweep is 10, 20, ..., 500 g: one mass per Isaac Lab
environment.  The program measures the downstream fixed-joint reaction at the
Axia80 center, records raw and no-load-tare-corrected CSV files, and creates a
force/moment response plot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

# Direct execution works before editable installation.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from axia80_feasibility.measurement import default_masses_g
from axia80_feasibility.model import build_model, repo_root


def arguments() -> argparse.Namespace:
    # AppLauncher must be imported and constructed before pxr/Isaac modules.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=PACKAGE_ROOT / "results/latest")
    parser.add_argument("--mass_start_g", type=int, default=10)
    parser.add_argument("--mass_stop_g", type=int, default=500)
    parser.add_argument("--mass_step_g", type=int, default=10)
    parser.add_argument("--env_spacing", type=float, default=2.0)
    parser.add_argument("--physics_hz", type=float, default=240.0)
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2].")
    parser.add_argument("--warmup_seconds", type=float, default=1.0)
    parser.add_argument("--tare_duration", type=float, default=0.75)
    parser.add_argument("--min_settle_seconds", type=float, default=1.0)
    parser.add_argument("--max_settle_seconds", type=float, default=8.0)
    parser.add_argument("--stable_window_seconds", type=float, default=0.5)
    parser.add_argument("--sample_duration", type=float, default=1.0)
    parser.add_argument("--linear_speed_threshold", type=float, default=0.005)
    parser.add_argument("--angular_speed_threshold", type=float, default=0.05)
    parser.add_argument("--height_tolerance", type=float, default=0.004)
    parser.add_argument("--drop_clearance", type=float, default=0.0005)
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--no_sample_csv", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless stability, alignment, and Fz linearity checks pass.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    try:
        args.masses_g = default_masses_g(args.mass_start_g, args.mass_stop_g, args.mass_step_g)
    except ValueError as error:
        parser.error(str(error))
    if len(args.masses_g) < 2:
        parser.error("The payload sweep requires at least two distinct masses for regression")
    positive = (
        "env_spacing",
        "physics_hz",
        "gravity",
        "warmup_seconds",
        "tare_duration",
        "min_settle_seconds",
        "max_settle_seconds",
        "stable_window_seconds",
        "sample_duration",
        "linear_speed_threshold",
        "angular_speed_threshold",
        "height_tolerance",
    )
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0 for name in positive):
        parser.error("All timing, threshold, gravity, frequency, and spacing options must be finite and positive")
    if not math.isfinite(args.drop_clearance) or args.drop_clearance < 0.0:
        parser.error("--drop_clearance must be finite and non-negative")
    if args.max_settle_seconds < args.min_settle_seconds:
        parser.error("--max_settle_seconds must be >= --min_settle_seconds")
    return args


def run(args: argparse.Namespace, app, urdf_path: Path, model: dict[str, object]) -> bool:
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.scene import InteractiveScene

    from axia80_feasibility.measurement import (
        evaluate_acceptance,
        regression_metrics,
        summarize_wrenches,
        write_sample_csv,
        write_summary_csv,
    )
    from axia80_feasibility.plotting import plot_summary
    from axia80_feasibility.scene import make_scene

    masses_g = list(args.masses_g)
    num_envs = len(masses_g)
    dt = 1.0 / args.physics_hz
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt,
            device=args.device,
            gravity=(0.0, 0.0, -args.gravity),
            render_interval=4,
            physx=sim_utils.PhysxCfg(
                solve_articulation_contact_last=True,
                bounce_threshold_velocity=0.05,
            ),
        )
    )
    scene = InteractiveScene(
        make_scene(
            urdf_path,
            model,
            num_envs=num_envs,
            env_spacing=args.env_spacing,
            force_conversion=args.force_conversion,
        )
    )

    # Imported UR collision subtrees may remain instance proxies.  Expand them
    # before simulation and set millimetre-scale contact offsets on the actual
    # tool collider so light payloads are not dominated by a large contact gap.
    for env_id in range(num_envs):
        robot_root = sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
        for prim in list(Usd.PrimRange(robot_root)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        sim_utils.modify_collision_properties(
            f"/World/envs/env_{env_id}/Robot/{model['tool_link_name']}",
            sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
        )
        for body_name in (model["sensor_link_name"], model["tool_link_name"]):
            prim = sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/{body_name}")
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    f"Rigid body '{body_name}' is missing in env_{env_id}; merge_fixed_joints must remain False"
                )

    sim.reset()
    robot = scene["robot"]
    payload = scene["payload"]
    robot.reset()
    payload.reset()
    # Simulation reset initializes PhysX views but does not guarantee that the
    # configured articulation pose has been authored into joint state.  Write
    # it explicitly so the first physics step starts with the broad tool face
    # horizontal instead of driving there from the all-zero URDF pose.
    joint_target = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_target, robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()

    tool_ids, tool_names = robot.find_bodies(str(model["tool_link_name"]), preserve_order=True)
    sensor_ids, sensor_names = robot.find_bodies(str(model["sensor_link_name"]), preserve_order=True)
    if len(tool_ids) != 1 or len(sensor_ids) != 1:
        raise RuntimeError(
            f"Expected one sensor and one tool body, got sensor={sensor_names}, tool={tool_names}; "
            f"available={robot.body_names}"
        )
    tool_body_id = tool_ids[0]
    sensor_body_id = sensor_ids[0]
    if robot.num_instances != num_envs or payload.num_instances != num_envs:
        raise RuntimeError(
            f"Vectorization mismatch: expected {num_envs}, robot={robot.num_instances}, payload={payload.num_instances}"
        )

    env_ids_cpu = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    masses_cpu = torch.tensor(masses_g, dtype=torch.float32, device="cpu").unsqueeze(-1) * 1.0e-3
    original_masses = payload.root_physx_view.get_masses().clone()
    inertias = payload.root_physx_view.get_inertias().clone()
    mass_ratio = masses_cpu / original_masses
    inertias = payload.data.default_inertia.cpu() * mass_ratio
    payload.root_physx_view.set_masses(masses_cpu, env_ids_cpu)
    payload.root_physx_view.set_inertias(inertias, env_ids_cpu)
    applied_masses = payload.root_physx_view.get_masses().cpu()
    if not torch.allclose(applied_masses, masses_cpu, rtol=1.0e-5, atol=1.0e-7):
        raise RuntimeError(
            f"PhysX payload masses do not match commands: commanded={masses_cpu.flatten().tolist()}, "
            f"applied={applied_masses.flatten().tolist()}"
        )

    # Keep payloads parked and gravity-free during the no-load tare window.
    gravity_disabled = torch.ones((num_envs, 1), dtype=torch.uint8, device="cpu")
    payload.root_physx_view.set_disable_gravities(gravity_disabled, env_ids_cpu)
    parked_state = payload.data.default_root_state.clone()
    parked_state[:, :3] += scene.env_origins
    parked_state[:, 7:] = 0.0
    payload.write_root_pose_to_sim(parked_state[:, :7])
    payload.write_root_velocity_to_sim(parked_state[:, 7:])

    def step_simulation() -> None:
        robot.set_joint_position_target(joint_target)
        scene.write_data_to_sim()
        sim.step(render=not args.headless)
        scene.update(dt)

    def reaction_wrench_sensor_frame() -> torch.Tensor:
        # PhysX supplies the parent-to-child incoming fixed-joint reaction.  In
        # this upward-facing assembly it already has the desired convention: a
        # centered downward payload produces positive sensor Fz.
        # The sensor parent and tool child link frames share the exact same
        # origin and orientation, so the known Isaac Lab/PhysX frame-doc
        # discrepancy cannot change these six reported components.
        return robot.data.body_incoming_joint_wrench_b[:, tool_body_id, :]

    warmup_steps = max(1, round(args.warmup_seconds / dt))
    for _ in range(warmup_steps):
        step_simulation()

    sensor_to_tool_position = torch.linalg.vector_norm(
        robot.data.body_pos_w[:, sensor_body_id] - robot.data.body_pos_w[:, tool_body_id], dim=-1
    )
    sensor_tool_quat_dot = torch.abs(
        torch.sum(
            robot.data.body_quat_w[:, sensor_body_id] * robot.data.body_quat_w[:, tool_body_id], dim=-1
        )
    )
    if float(sensor_to_tool_position.max()) > 1.0e-4 or float(sensor_tool_quat_dot.min()) < 0.999999:
        raise RuntimeError(
            "Sensor and tool measurement frames do not coincide: "
            f"max position error={float(sensor_to_tool_position.max()):.6g} m, "
            f"min |quat dot|={float(sensor_tool_quat_dot.min()):.9f}"
        )

    local_up = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
    local_up[:, 2] = 1.0
    tool_normal_w = math_utils.quat_apply(robot.data.body_quat_w[:, tool_body_id], local_up)
    if float(tool_normal_w[:, 2].min()) < 0.999:
        raise RuntimeError(
            "Tool broad-face +normal is not upward: "
            f"minimum world-Z component={float(tool_normal_w[:, 2].min()):.6f}, "
            f"env0 normal={tool_normal_w[0].detach().cpu().tolist()}, "
            f"joint_names={robot.joint_names}, "
            f"env0 joint_pos={robot.data.joint_pos[0].detach().cpu().tolist()}"
        )

    first_origin = scene.env_origins[0].detach().cpu().tolist()
    sim.set_camera_view(
        eye=[first_origin[0] + 1.25, first_origin[1] + 1.0, 1.05],
        target=[first_origin[0] + 0.49, first_origin[1] + 0.13, 0.70],
    )
    print(
        f"[INFO]: {num_envs} vectorized environments, masses={masses_g[0]}..{masses_g[-1]} g "
        f"in {args.mass_step_g} g increments"
    )
    print(
        f"[INFO]: Axia80 cylinder radius={float(model['sensor_radius_m']):.4f} m, "
        f"height={float(model['sensor_height_m']):.4f} m"
    )
    print(
        f"[INFO]: wrench=[Fx,Fy,Fz,Mx,My,Mz] at sensor center; "
        f"tool normal min z={float(tool_normal_w[:, 2].min()):.7f}"
    )

    tare_steps = max(2, round(args.tare_duration / dt))
    tare_samples_device: list[torch.Tensor] = []
    for _ in range(tare_steps):
        step_simulation()
        tare_samples_device.append(reaction_wrench_sensor_frame().clone())
    tare_samples_tensor = torch.stack(tare_samples_device)
    if not bool(torch.isfinite(tare_samples_tensor).all()):
        raise RuntimeError("Non-finite wrench detected during the tare window")

    # Place every payload at the center of the broad upper tool face.  The link
    # pose is the Axia80 center; the scalar offset is therefore exact and does
    # not rely on a hard-coded UR5e world pose.
    tool_normal_w = math_utils.quat_apply(robot.data.body_quat_w[:, tool_body_id], local_up)
    center_offset = (
        float(model["tool_top_z_in_sensor_m"])
        + 0.5 * float(model["payload_size_m"])
        + args.drop_clearance
    )
    payload_pose = payload.data.root_pose_w.clone()
    payload_pose[:, :3] = robot.data.body_pos_w[:, tool_body_id] + center_offset * tool_normal_w
    payload_pose[:, 3:7] = robot.data.body_quat_w[:, tool_body_id]
    payload_velocity = torch.zeros((num_envs, 6), dtype=torch.float32, device=sim.device)
    gravity_enabled = torch.zeros((num_envs, 1), dtype=torch.uint8, device="cpu")
    payload.root_physx_view.set_disable_gravities(gravity_enabled, env_ids_cpu)
    # Write state after re-enabling gravity so previously sleeping parked
    # bodies are awakened and enter the contact solve.
    payload.write_root_pose_to_sim(payload_pose)
    payload.write_root_velocity_to_sim(payload_velocity)
    payload.root_physx_view.wake_up(env_ids_cpu)
    applied_gravity_flags = payload.root_physx_view.get_disable_gravities().cpu()
    if bool(torch.any(applied_gravity_flags != 0)):
        raise RuntimeError(
            f"Payload gravity did not re-enable: flags={applied_gravity_flags.flatten().tolist()}"
        )

    minimum_steps = max(1, round(args.min_settle_seconds / dt))
    maximum_steps = max(minimum_steps, round(args.max_settle_seconds / dt))
    required_stable_steps = max(1, round(args.stable_window_seconds / dt))
    consecutive_stable = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
    settled_mask = torch.zeros(num_envs, dtype=torch.bool, device=sim.device)
    settle_steps = maximum_steps
    expected_height = float(model["tool_top_z_in_sensor_m"]) + 0.5 * float(model["payload_size_m"])
    lateral_limit = 0.5 * min(float(value) for value in model["tool_size_m"][:2]) - 0.5 * float(
        model["payload_size_m"]
    )

    def relative_payload_speed(relative_position_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Payload velocity relative to the corresponding moving tool point."""
        tool_velocity = robot.data.body_link_vel_w[:, tool_body_id]
        tool_point_linear_velocity = tool_velocity[:, :3] + torch.linalg.cross(
            tool_velocity[:, 3:], relative_position_w, dim=-1
        )
        relative_linear_velocity = payload.data.root_com_lin_vel_w - tool_point_linear_velocity
        relative_angular_velocity = payload.data.root_com_ang_vel_w - tool_velocity[:, 3:]
        return (
            torch.linalg.vector_norm(relative_linear_velocity, dim=-1),
            torch.linalg.vector_norm(relative_angular_velocity, dim=-1),
        )

    for settle_step in range(maximum_steps):
        step_simulation()
        relative = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        tool_normal_w = math_utils.quat_apply(robot.data.body_quat_w[:, tool_body_id], local_up)
        normal_distance = torch.sum(relative * tool_normal_w, dim=-1)
        lateral = relative - normal_distance.unsqueeze(-1) * tool_normal_w
        speed, angular_speed = relative_payload_speed(relative)
        stable_now = (
            (speed <= args.linear_speed_threshold)
            & (angular_speed <= args.angular_speed_threshold)
            & (torch.abs(normal_distance - expected_height) <= args.height_tolerance)
            & (torch.linalg.vector_norm(lateral, dim=-1) <= lateral_limit)
        )
        if settle_step + 1 < minimum_steps:
            stable_now.zero_()
        consecutive_stable = torch.where(stable_now, consecutive_stable + 1, torch.zeros_like(consecutive_stable))
        settled_mask = consecutive_stable >= required_stable_steps
        if settle_step % max(1, round(args.physics_hz)) == 0:
            print(
                f"[INFO]: settling t={(settle_step + 1) * dt:.2f}s, "
                f"stable={int(settled_mask.sum())}/{num_envs}, "
                f"max_speed={float(speed.max()):.6f} m/s"
            )
        if bool(settled_mask.all()):
            settle_steps = settle_step + 1
            break

    if not bool(settled_mask.all()):
        missing = torch.nonzero(~settled_mask, as_tuple=False).flatten().detach().cpu().tolist()
        raise RuntimeError(
            "Stability timeout: refusing to record measurements before every payload is stable; "
            f"unsettled environment ids={missing}"
        )
    else:
        print(f"[INFO]: all payloads stable after {settle_steps * dt:.3f} s")

    sample_steps = max(2, round(args.sample_duration / dt))
    samples_device: list[torch.Tensor] = []
    sample_stable_mask = torch.ones(num_envs, dtype=torch.bool, device=sim.device)
    sample_max_speed = torch.zeros(num_envs, dtype=torch.float32, device=sim.device)
    sample_max_angular_speed = torch.zeros_like(sample_max_speed)
    sample_min_tool_normal_z = torch.ones_like(sample_max_speed)
    for _ in range(sample_steps):
        step_simulation()
        wrench = reaction_wrench_sensor_frame().clone()
        if not bool(torch.isfinite(wrench).all()):
            raise RuntimeError("Non-finite wrench detected during the payload sample window")
        samples_device.append(wrench)
        relative = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        tool_normal_w = math_utils.quat_apply(robot.data.body_quat_w[:, tool_body_id], local_up)
        sample_min_tool_normal_z = torch.minimum(sample_min_tool_normal_z, tool_normal_w[:, 2])
        normal_distance = torch.sum(relative * tool_normal_w, dim=-1)
        lateral = relative - normal_distance.unsqueeze(-1) * tool_normal_w
        relative_speed, relative_angular_speed = relative_payload_speed(relative)
        sample_max_speed = torch.maximum(sample_max_speed, relative_speed)
        sample_max_angular_speed = torch.maximum(sample_max_angular_speed, relative_angular_speed)
        sample_stable_mask &= (
            (relative_speed <= args.linear_speed_threshold)
            & (relative_angular_speed <= args.angular_speed_threshold)
            & (torch.abs(normal_distance - expected_height) <= args.height_tolerance)
            & (torch.linalg.vector_norm(lateral, dim=-1) <= lateral_limit)
        )
    samples_tensor = torch.stack(samples_device)

    if not bool(sample_stable_mask.all()):
        missing = torch.nonzero(~sample_stable_mask, as_tuple=False).flatten().detach().cpu().tolist()
        raise RuntimeError(
            "Payload stability was lost during the measurement window; refusing to write CSV data; "
            f"environment ids={missing}"
        )
    settled_mask &= sample_stable_mask
    samples = samples_tensor.detach().cpu().numpy()
    tare_samples = tare_samples_tensor.detach().cpu().numpy()
    rows = summarize_wrenches(
        masses_g,
        samples,
        tare_samples,
        gravity_m_s2=args.gravity,
        settled=settled_mask.detach().cpu().tolist(),
        settle_time_s=settle_steps * dt,
        sample_max_relative_speed_m_s=sample_max_speed.detach().cpu().tolist(),
        sample_max_relative_angular_speed_rad_s=sample_max_angular_speed.detach().cpu().tolist(),
        tool_normal_z=sample_min_tool_normal_z.detach().cpu().tolist(),
    )
    metrics = regression_metrics(
        rows,
        expected_tare_fz_n=float(model["tool_mass_kg"]) * args.gravity,
    )
    passed, failures = evaluate_acceptance(metrics, gravity_m_s2=args.gravity)

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "axia80_payload_summary.csv"
    samples_path = output_dir / "axia80_payload_samples.csv"
    plot_path = output_dir / "axia80_payload_response.png"
    metadata_path = output_dir / "axia80_payload_metadata.json"
    write_summary_csv(summary_path, rows)
    if not args.no_sample_csv:
        write_sample_csv(
            samples_path,
            masses_g,
            samples,
            np.mean(tare_samples, axis=0),
            dt_s=dt,
        )
    plot_summary(summary_path, plot_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
        "metrics": metrics,
        "masses_g": masses_g,
        "num_envs": num_envs,
        "simulation_dt_s": dt,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": model,
        "wrench_convention": {
            "order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "frame": "Axia80 center; +Z is upward broad-face normal",
            "sign": "PhysX parent-to-child incoming tool-joint reaction; centered downward payload yields +Fz",
            "reported_primary_columns": "no-load tare corrected",
        },
        "artifacts": {
            "summary_csv": str(summary_path),
            "sample_csv": None if args.no_sample_csv else str(samples_path),
            "plot_png": str(plot_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[RESULT]: {'PASS' if passed else 'FAIL'}")
    print("[RESULT]: " + json.dumps(metrics, sort_keys=True))
    for failure in failures:
        print(f"[RESULT]: check failed: {failure}")
    print(f"[RESULT]: summary CSV: {summary_path}")
    if not args.no_sample_csv:
        print(f"[RESULT]: sample CSV:  {samples_path}")
    print(f"[RESULT]: plot:       {plot_path}")
    print(f"[RESULT]: metadata:   {metadata_path}")
    return passed


def main() -> None:
    args = arguments()
    root = (args.repo_root or repo_root()).resolve()
    urdf_path, model = build_model(root, PACKAGE_ROOT / "generated")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    exit_code = 0
    try:
        passed = run(args, launcher.app, urdf_path, model)
        if args.strict and not passed:
            exit_code = 2
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Artifacts are written synchronously before shutdown.  This avoids the
        # known Isaac Sim 5.1 GPU stage-close deadlock in standalone processes.
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
