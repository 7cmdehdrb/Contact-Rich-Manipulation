"""Run the independent wrist_3 +10-degree high-friction payload experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from axia80_feasibility.measurement import default_masses_g
from axia80_feasibility.tilted_cantilever_model import (
    build_tilted_cantilever_model,
    repo_root,
)


def arguments() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PACKAGE_ROOT / "results/tilted_cantilever_10deg_latest",
    )
    parser.add_argument("--mass_start_g", type=int, default=10)
    parser.add_argument("--mass_stop_g", type=int, default=500)
    parser.add_argument("--mass_step_g", type=int, default=10)
    parser.add_argument("--env_spacing", type=float, default=2.0)
    parser.add_argument("--physics_hz", type=float, default=240.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--warmup_seconds", type=float, default=1.0)
    parser.add_argument("--tare_duration", type=float, default=0.75)
    parser.add_argument("--min_settle_seconds", type=float, default=1.0)
    parser.add_argument("--max_settle_seconds", type=float, default=8.0)
    parser.add_argument("--stable_window_seconds", type=float, default=0.5)
    parser.add_argument("--sample_duration", type=float, default=1.0)
    parser.add_argument("--linear_speed_threshold", type=float, default=0.005)
    parser.add_argument("--angular_speed_threshold", type=float, default=0.05)
    parser.add_argument("--position_tolerance", type=float, default=0.004)
    parser.add_argument("--slide_tolerance", type=float, default=0.004)
    parser.add_argument("--orientation_tolerance_deg", type=float, default=2.0)
    parser.add_argument("--drop_clearance", type=float, default=0.0005)
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--no_sample_csv", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless distribution, no-slide, stability, and pose checks pass.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    try:
        args.masses_g = default_masses_g(
            args.mass_start_g,
            args.mass_stop_g,
            args.mass_step_g,
        )
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
        "position_tolerance",
        "slide_tolerance",
        "orientation_tolerance_deg",
    )
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0
        for name in positive
    ):
        parser.error("All timing, threshold, gravity, frequency, and spacing options must be positive")
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

    from axia80_feasibility.measurement import write_sample_csv, write_summary_csv
    from axia80_feasibility.tilted_cantilever_measurement import (
        evaluate_tilted_acceptance,
        expected_payload_wrench_per_kg,
        expected_tool_tare_wrench,
        summarize_tilted_wrenches,
        tilted_metrics,
    )
    from axia80_feasibility.tilted_cantilever_plotting import (
        plot_tilted_cantilever_summary,
    )
    from axia80_feasibility.tilted_cantilever_scene import make_tilted_scene

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
        make_tilted_scene(
            urdf_path,
            model,
            num_envs=num_envs,
            env_spacing=args.env_spacing,
            force_conversion=args.force_conversion,
        )
    )

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
                    f"Rigid body '{body_name}' is missing in env_{env_id}; "
                    "merge_fixed_joints must remain False"
                )

    sim.reset()
    robot = scene["robot"]
    payload = scene["payload"]
    robot.reset()
    payload.reset()
    joint_target = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_target, robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()

    tool_ids, tool_names = robot.find_bodies(str(model["tool_link_name"]), preserve_order=True)
    sensor_ids, sensor_names = robot.find_bodies(
        str(model["sensor_link_name"]),
        preserve_order=True,
    )
    if len(tool_ids) != 1 or len(sensor_ids) != 1:
        raise RuntimeError(
            f"Expected one sensor and one tool body, got sensor={sensor_names}, "
            f"tool={tool_names}; available={robot.body_names}"
        )
    tool_body_id = tool_ids[0]
    sensor_body_id = sensor_ids[0]
    if robot.num_instances != num_envs or payload.num_instances != num_envs:
        raise RuntimeError(
            f"Vectorization mismatch: expected {num_envs}, robot={robot.num_instances}, "
            f"payload={payload.num_instances}"
        )

    env_ids_cpu = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    masses_cpu = (
        torch.tensor(masses_g, dtype=torch.float32, device="cpu").unsqueeze(-1) * 1.0e-3
    )
    original_masses = payload.root_physx_view.get_masses().clone()
    mass_ratio = masses_cpu / original_masses
    inertias = payload.data.default_inertia.cpu() * mass_ratio
    payload.root_physx_view.set_masses(masses_cpu, env_ids_cpu)
    payload.root_physx_view.set_inertias(inertias, env_ids_cpu)
    applied_masses = payload.root_physx_view.get_masses().cpu()
    if not torch.allclose(applied_masses, masses_cpu, rtol=1.0e-5, atol=1.0e-7):
        raise RuntimeError(
            f"PhysX payload masses differ from commands: "
            f"commanded={masses_cpu.flatten().tolist()}, "
            f"applied={applied_masses.flatten().tolist()}"
        )

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
        return robot.data.body_incoming_joint_wrench_b[:, tool_body_id, :]

    warmup_steps = max(1, round(args.warmup_seconds / dt))
    for _ in range(warmup_steps):
        step_simulation()

    sensor_to_tool_position = torch.linalg.vector_norm(
        robot.data.body_pos_w[:, sensor_body_id] - robot.data.body_pos_w[:, tool_body_id],
        dim=-1,
    )
    sensor_tool_quat_dot = torch.abs(
        torch.sum(
            robot.data.body_quat_w[:, sensor_body_id]
            * robot.data.body_quat_w[:, tool_body_id],
            dim=-1,
        )
    )
    if float(sensor_to_tool_position.max()) > 1.0e-4 or float(sensor_tool_quat_dot.min()) < 0.999999:
        raise RuntimeError(
            "Sensor and tool measurement frames do not coincide: "
            f"max position error={float(sensor_to_tool_position.max()):.6g} m, "
            f"min |quat dot|={float(sensor_tool_quat_dot.min()):.9f}"
        )

    local_broad_normal = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
    local_broad_normal[:, 0] = 1.0
    local_downhill_axis = torch.zeros_like(local_broad_normal)
    local_downhill_axis[:, 1] = 1.0
    local_cantilever_axis = torch.zeros_like(local_broad_normal)
    local_cantilever_axis[:, 2] = 1.0
    tool_quat = robot.data.body_quat_w[:, tool_body_id]
    broad_normal_w = math_utils.quat_apply(tool_quat, local_broad_normal)
    downhill_axis_w = math_utils.quat_apply(tool_quat, local_downhill_axis)
    cantilever_axis_w = math_utils.quat_apply(tool_quat, local_cantilever_axis)
    expected_normal_z = math.cos(float(model["wrist3_tilt_rad"]))
    expected_downhill_z = -math.sin(float(model["wrist3_tilt_rad"]))
    # The implicit arm drives yield a few tenths of a degree under the tool's
    # own bending load.  Reject gross pose errors here, while the measured
    # force/moment distribution below checks the effective angle precisely.
    initial_axis_tolerance = 0.01
    if (
        float(torch.max(torch.abs(broad_normal_w[:, 2] - expected_normal_z)))
        > initial_axis_tolerance
    ):
        raise RuntimeError(
            "Tool broad-face normal does not match the configured 10-degree tilt: "
            f"actual env0={broad_normal_w[0].detach().cpu().tolist()}, "
            f"expected world-Z={expected_normal_z:.9f}"
        )
    if (
        float(torch.max(torch.abs(downhill_axis_w[:, 2] - expected_downhill_z)))
        > initial_axis_tolerance
    ):
        raise RuntimeError(
            "Tool local +Y does not point down the configured incline: "
            f"actual env0={downhill_axis_w[0].detach().cpu().tolist()}, "
            f"expected world-Z={expected_downhill_z:.9f}, "
            f"joint_pos={robot.data.joint_pos[0].detach().cpu().tolist()}"
        )
    if float(torch.abs(cantilever_axis_w[:, 2]).max()) > 0.001:
        raise RuntimeError("Cantilever/sensor +Z axis is not horizontal")

    first_sensor = robot.data.body_pos_w[0, sensor_body_id].detach().cpu().tolist()
    sim.set_camera_view(
        eye=[first_sensor[0] + 0.9, first_sensor[1] + 0.7, first_sensor[2] + 0.5],
        target=[first_sensor[0], first_sensor[1] - 0.12, first_sensor[2]],
    )
    friction_ratio = float(model["static_friction"]) / float(
        model["minimum_static_friction_for_no_slide"]
    )
    print(
        f"[INFO]: {num_envs} tilted cantilever environments, "
        f"masses={masses_g[0]}..{masses_g[-1]} g in {args.mass_step_g} g increments"
    )
    print(
        f"[INFO]: wrist_3 tilt=+{float(model['wrist3_tilt_deg']):.1f} deg, "
        f"tool size={model['tool_size_sensor_frame_m']} m"
    )
    print(
        f"[INFO]: payload friction static/dynamic="
        f"{float(model['static_friction']):.2f}/{float(model['dynamic_friction']):.2f}, "
        f"static safety ratio={friction_ratio:.2f}x over tan(tilt)"
    )
    print(
        "[INFO]: expected distributed channels: +Fx, -Fy, +Mx, +My, -Mz; Fz approximately zero"
    )

    tare_steps = max(2, round(args.tare_duration / dt))
    tare_samples_device: list[torch.Tensor] = []
    for _ in range(tare_steps):
        step_simulation()
        tare_samples_device.append(reaction_wrench_sensor_frame().clone())
    tare_samples_tensor = torch.stack(tare_samples_device)
    if not bool(torch.isfinite(tare_samples_tensor).all()):
        raise RuntimeError("Non-finite wrench detected during the tare window")

    payload_center_local = torch.tensor(
        model["payload_center_in_sensor_m"],
        dtype=torch.float32,
        device=sim.device,
    ).repeat(num_envs, 1)
    drop_center_local = payload_center_local.clone()
    drop_center_local[:, 0] += args.drop_clearance
    payload_pose = payload.data.root_pose_w.clone()
    payload_pose[:, :3] = robot.data.body_pos_w[:, tool_body_id] + math_utils.quat_apply(
        robot.data.body_quat_w[:, tool_body_id],
        drop_center_local,
    )
    payload_pose[:, 3:7] = robot.data.body_quat_w[:, tool_body_id]
    payload_velocity = torch.zeros((num_envs, 6), dtype=torch.float32, device=sim.device)
    gravity_enabled = torch.zeros((num_envs, 1), dtype=torch.uint8, device="cpu")
    payload.root_physx_view.set_disable_gravities(gravity_enabled, env_ids_cpu)
    payload.write_root_pose_to_sim(payload_pose)
    payload.write_root_velocity_to_sim(payload_velocity)
    payload.root_physx_view.wake_up(env_ids_cpu)
    applied_gravity_flags = payload.root_physx_view.get_disable_gravities().cpu()
    if bool(torch.any(applied_gravity_flags != 0)):
        raise RuntimeError(
            f"Payload gravity did not re-enable: "
            f"flags={applied_gravity_flags.flatten().tolist()}"
        )

    tool_half_width = 0.5 * float(model["tool_size_sensor_frame_m"][1])
    payload_half_width = 0.5 * float(model["payload_size_m"])
    payload_masses_device = masses_cpu.flatten().to(device=sim.device)

    def expected_payload_wrench_from_pose() -> torch.Tensor:
        """Resolve gravity and r x F using each sampled tool/payload pose."""
        current_tool_quat = robot.data.body_quat_w[:, tool_body_id]
        relative_w = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        relative_local = math_utils.quat_apply_inverse(current_tool_quat, relative_w)
        support_force_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
        support_force_w[:, 2] = payload_masses_device * args.gravity
        support_force_local = math_utils.quat_apply_inverse(
            current_tool_quat,
            support_force_w,
        )
        support_moment_local = torch.linalg.cross(
            relative_local,
            support_force_local,
            dim=-1,
        )
        return torch.cat((support_force_local, support_moment_local), dim=-1)

    def kinematics() -> tuple[torch.Tensor, ...]:
        relative = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        current_tool_quat = robot.data.body_quat_w[:, tool_body_id]
        relative_local = math_utils.quat_apply_inverse(current_tool_quat, relative)
        position_error = torch.linalg.vector_norm(relative_local - payload_center_local, dim=-1)
        downhill_displacement = torch.abs(relative_local[:, 1])
        edge_margin = tool_half_width - payload_half_width - downhill_displacement
        payload_quat = payload.data.root_pose_w[:, 3:7]
        quat_dot = torch.abs(torch.sum(payload_quat * current_tool_quat, dim=-1))
        relative_angle_deg = torch.rad2deg(
            2.0 * torch.acos(torch.clamp(quat_dot, min=0.0, max=1.0))
        )
        tool_velocity = robot.data.body_link_vel_w[:, tool_body_id]
        tool_point_linear_velocity = tool_velocity[:, :3] + torch.linalg.cross(
            tool_velocity[:, 3:],
            relative,
            dim=-1,
        )
        relative_linear_velocity = payload.data.root_com_lin_vel_w - tool_point_linear_velocity
        relative_angular_velocity = payload.data.root_com_ang_vel_w - tool_velocity[:, 3:]
        speed = torch.linalg.vector_norm(relative_linear_velocity, dim=-1)
        angular_speed = torch.linalg.vector_norm(relative_angular_velocity, dim=-1)
        normal = math_utils.quat_apply(current_tool_quat, local_broad_normal)
        axis = math_utils.quat_apply(current_tool_quat, local_cantilever_axis)
        return (
            position_error,
            downhill_displacement,
            edge_margin,
            relative_angle_deg,
            speed,
            angular_speed,
            normal[:, 2],
            torch.abs(axis[:, 2]),
        )

    minimum_steps = max(1, round(args.min_settle_seconds / dt))
    maximum_steps = max(minimum_steps, round(args.max_settle_seconds / dt))
    required_stable_steps = max(1, round(args.stable_window_seconds / dt))
    consecutive_stable = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
    settled_mask = torch.zeros(num_envs, dtype=torch.bool, device=sim.device)
    settle_steps = maximum_steps
    for settle_step in range(maximum_steps):
        step_simulation()
        (
            position_error,
            downhill_displacement,
            edge_margin,
            relative_angle_deg,
            speed,
            angular_speed,
            _,
            _,
        ) = kinematics()
        stable_now = (
            (speed <= args.linear_speed_threshold)
            & (angular_speed <= args.angular_speed_threshold)
            & (position_error <= args.position_tolerance)
            & (downhill_displacement <= args.slide_tolerance)
            & (relative_angle_deg <= args.orientation_tolerance_deg)
            & (edge_margin > 0.0)
        )
        if settle_step + 1 < minimum_steps:
            stable_now.zero_()
        consecutive_stable = torch.where(
            stable_now,
            consecutive_stable + 1,
            torch.zeros_like(consecutive_stable),
        )
        settled_mask = consecutive_stable >= required_stable_steps
        if settle_step % max(1, round(args.physics_hz)) == 0:
            print(
                f"[INFO]: settling t={(settle_step + 1) * dt:.2f}s, "
                f"stable={int(settled_mask.sum())}/{num_envs}, "
                f"max_speed={float(speed.max()):.6f} m/s, "
                f"max_slide={float(downhill_displacement.max()):.6f} m, "
                f"max_tilt_error={float(relative_angle_deg.max()):.4f} deg"
            )
        if bool(settled_mask.all()):
            settle_steps = settle_step + 1
            break

    if not bool(settled_mask.all()):
        missing = torch.nonzero(~settled_mask, as_tuple=False).flatten().detach().cpu().tolist()
        raise RuntimeError(
            "Stability/no-slide timeout: refusing to record measurements; "
            f"unsettled environment ids={missing}"
        )
    print(f"[INFO]: all tilted payloads stable after {settle_steps * dt:.3f} s")

    sample_steps = max(2, round(args.sample_duration / dt))
    samples_device: list[torch.Tensor] = []
    expected_samples_device: list[torch.Tensor] = []
    sample_stable_mask = torch.ones(num_envs, dtype=torch.bool, device=sim.device)
    sample_max_speed = torch.zeros(num_envs, dtype=torch.float32, device=sim.device)
    sample_max_angular_speed = torch.zeros_like(sample_max_speed)
    sample_max_position_error = torch.zeros_like(sample_max_speed)
    sample_max_downhill = torch.zeros_like(sample_max_speed)
    sample_max_relative_angle = torch.zeros_like(sample_max_speed)
    sample_min_edge_margin = torch.full_like(sample_max_speed, math.inf)
    sample_min_broad_normal_z = torch.ones_like(sample_max_speed)
    sample_max_broad_normal_z = torch.full_like(sample_max_speed, -math.inf)
    sample_max_abs_sensor_axis_z = torch.zeros_like(sample_max_speed)
    for _ in range(sample_steps):
        step_simulation()
        wrench = reaction_wrench_sensor_frame().clone()
        if not bool(torch.isfinite(wrench).all()):
            raise RuntimeError("Non-finite wrench detected during the payload sample window")
        samples_device.append(wrench)
        expected_samples_device.append(expected_payload_wrench_from_pose())
        (
            position_error,
            downhill_displacement,
            edge_margin,
            relative_angle_deg,
            speed,
            angular_speed,
            normal_z,
            axis_abs_z,
        ) = kinematics()
        sample_max_speed = torch.maximum(sample_max_speed, speed)
        sample_max_angular_speed = torch.maximum(sample_max_angular_speed, angular_speed)
        sample_max_position_error = torch.maximum(sample_max_position_error, position_error)
        sample_max_downhill = torch.maximum(sample_max_downhill, downhill_displacement)
        sample_max_relative_angle = torch.maximum(sample_max_relative_angle, relative_angle_deg)
        sample_min_edge_margin = torch.minimum(sample_min_edge_margin, edge_margin)
        sample_min_broad_normal_z = torch.minimum(sample_min_broad_normal_z, normal_z)
        sample_max_broad_normal_z = torch.maximum(sample_max_broad_normal_z, normal_z)
        sample_max_abs_sensor_axis_z = torch.maximum(
            sample_max_abs_sensor_axis_z,
            axis_abs_z,
        )
        sample_stable_mask &= (
            (speed <= args.linear_speed_threshold)
            & (angular_speed <= args.angular_speed_threshold)
            & (position_error <= args.position_tolerance)
            & (downhill_displacement <= args.slide_tolerance)
            & (relative_angle_deg <= args.orientation_tolerance_deg)
            & (edge_margin > 0.0)
        )
    samples_tensor = torch.stack(samples_device)
    expected_samples_tensor = torch.stack(expected_samples_device)
    if not bool(sample_stable_mask.all()):
        missing = (
            torch.nonzero(~sample_stable_mask, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        raise RuntimeError(
            "Payload slid, tipped, or lost stability during measurement; "
            f"refusing to write CSV data; environment ids={missing}"
        )

    settled_mask &= sample_stable_mask
    samples = samples_tensor.detach().cpu().numpy()
    expected_samples = expected_samples_tensor.detach().cpu().numpy()
    tare_samples = tare_samples_tensor.detach().cpu().numpy()
    lever_arm = float(model["payload_bending_lever_arm_m"])
    payload_normal_offset = float(model["payload_center_in_sensor_m"][0])
    tilt_rad = float(model["wrist3_tilt_rad"])
    rows = summarize_tilted_wrenches(
        masses_g,
        samples,
        tare_samples,
        gravity_m_s2=args.gravity,
        tilt_rad=tilt_rad,
        bending_lever_arm_m=lever_arm,
        payload_normal_offset_m=payload_normal_offset,
        expected_samples=expected_samples,
        settled=settled_mask.detach().cpu().tolist(),
        settle_time_s=settle_steps * dt,
        sample_max_relative_speed_m_s=sample_max_speed.detach().cpu().tolist(),
        sample_max_relative_angular_speed_rad_s=(
            sample_max_angular_speed.detach().cpu().tolist()
        ),
        sample_max_position_error_m=sample_max_position_error.detach().cpu().tolist(),
        sample_max_abs_downhill_displacement_m=(
            sample_max_downhill.detach().cpu().tolist()
        ),
        sample_max_payload_tool_angle_deg=(
            sample_max_relative_angle.detach().cpu().tolist()
        ),
        sample_min_edge_margin_m=sample_min_edge_margin.detach().cpu().tolist(),
        sample_min_broad_normal_z=sample_min_broad_normal_z.detach().cpu().tolist(),
        sample_max_broad_normal_z=sample_max_broad_normal_z.detach().cpu().tolist(),
        sample_max_abs_sensor_axis_z=(
            sample_max_abs_sensor_axis_z.detach().cpu().tolist()
        ),
    )
    expected_per_kg = expected_payload_wrench_per_kg(
        gravity_m_s2=args.gravity,
        tilt_rad=tilt_rad,
        bending_lever_arm_m=lever_arm,
        payload_normal_offset_m=payload_normal_offset,
    )
    expected_tare = expected_tool_tare_wrench(
        tool_mass_kg=float(model["tool_mass_kg"]),
        gravity_m_s2=args.gravity,
        tilt_rad=tilt_rad,
        tool_center_z_m=float(model["tool_center_z_in_sensor_m"]),
    )
    metrics = tilted_metrics(rows, expected_tare=expected_tare)
    passed, failures = evaluate_tilted_acceptance(
        metrics,
        expected_slopes=expected_per_kg,
        position_tolerance_m=min(args.position_tolerance, args.slide_tolerance),
        max_payload_tool_angle_deg=args.orientation_tolerance_deg,
    )

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "axia80_tilted_cantilever_summary.csv"
    samples_path = output_dir / "axia80_tilted_cantilever_samples.csv"
    plot_path = output_dir / "axia80_tilted_cantilever_response.png"
    metadata_path = output_dir / "axia80_tilted_cantilever_metadata.json"
    run_log_path = output_dir / "axia80_tilted_cantilever_run.log"
    write_summary_csv(summary_path, rows)
    if not args.no_sample_csv:
        write_sample_csv(
            samples_path,
            masses_g,
            samples,
            np.mean(tare_samples, axis=0),
            dt_s=dt,
        )
    plot_tilted_cantilever_summary(summary_path, plot_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary_csv": str(summary_path),
        "sample_csv": None if args.no_sample_csv else str(samples_path),
        "plot_png": str(plot_path),
        "run_log": str(run_log_path),
    }
    metadata = {
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
        "metrics": metrics,
        "expected_payload_wrench_per_kg": expected_per_kg.tolist(),
        "expected_tool_tare_wrench": expected_tare.tolist(),
        "masses_g": masses_g,
        "num_envs": num_envs,
        "simulation_dt_s": dt,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": model,
        "wrench_convention": {
            "order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "frame": "Axia80 center; wrist_3 is +10 degrees about sensor +Z",
            "expected_distribution": "+Fx, -Fy, +Mx, +My, -Mz; Fz approximately zero",
            "reported_primary_columns": "tool-only tare corrected",
        },
        "anti_slide_checks": {
            "static_friction": float(model["static_friction"]),
            "dynamic_friction": float(model["dynamic_friction"]),
            "minimum_static_friction": float(
                model["minimum_static_friction_for_no_slide"]
            ),
            "friction_safety_ratio": friction_ratio,
            "slide_tolerance_m": args.slide_tolerance,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
        },
        "artifacts": artifacts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    run_log_lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        "experiment=UR5e-Axia80 wrist3 +10deg high-friction cantilever sweep",
        f"verdict={'PASS' if passed else 'FAIL'}",
        f"num_envs={num_envs}",
        f"mass_range_g={masses_g[0]}..{masses_g[-1]} step={args.mass_step_g}",
        f"settle_time_s={settle_steps * dt:.9f}",
        f"tool_width_m={float(model['tool_size_sensor_frame_m'][1]):.9f}",
        f"wrist3_tilt_deg={float(model['wrist3_tilt_deg']):.9f}",
        f"static_friction={float(model['static_friction']):.9f}",
        f"dynamic_friction={float(model['dynamic_friction']):.9f}",
        f"friction_safety_ratio={friction_ratio:.9f}",
        "expected_payload_wrench_per_kg=" + json.dumps(expected_per_kg.tolist()),
        "metrics=" + json.dumps(metrics, sort_keys=True),
        "failures=" + json.dumps(failures),
        "artifacts=" + json.dumps(artifacts, sort_keys=True),
    ]
    run_log_path.write_text("\n".join(run_log_lines) + "\n")

    print(f"[RESULT]: {'PASS' if passed else 'FAIL'}")
    print("[RESULT]: " + json.dumps(metrics, sort_keys=True))
    for failure in failures:
        print(f"[RESULT]: check failed: {failure}")
    print(f"[RESULT]: summary CSV: {summary_path}")
    if not args.no_sample_csv:
        print(f"[RESULT]: sample CSV:  {samples_path}")
    print(f"[RESULT]: plot:       {plot_path}")
    print(f"[RESULT]: metadata:   {metadata_path}")
    print(f"[RESULT]: run log:    {run_log_path}")
    return passed


def main() -> None:
    args = arguments()
    root = (args.repo_root or repo_root()).resolve()
    urdf_path, model = build_tilted_cantilever_model(
        root,
        PACKAGE_ROOT / "generated/tilted_cantilever_10deg",
    )
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
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
