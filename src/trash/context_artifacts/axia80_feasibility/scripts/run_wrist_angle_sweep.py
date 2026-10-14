"""Run a fixed-mass vectorized wrist Roll x Tilt Axia80 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from axia80_feasibility.wrist_angle_model import (
    DEFAULT_PAYLOAD_MASS_G,
    DEFAULT_ROLL_ANGLES_DEG,
    DEFAULT_TILT_ANGLES_DEG,
    angle_grid,
    build_wrist_angle_model,
    repo_root,
)


ARTIFACT_NAMES = (
    "axia80_wrist_angle_summary.csv",
    "axia80_wrist_angle_samples.csv",
    "axia80_wrist_angle_response_heatmaps.png",
    "axia80_wrist_angle_error_heatmaps.png",
    "axia80_wrist_angle_center_slices.png",
    "axia80_wrist_angle_metadata.json",
    "axia80_wrist_angle_run.log",
)


def arguments() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PACKAGE_ROOT / "results/wrist_angle_sweep_latest",
    )
    parser.add_argument("--payload_mass_g", type=int, default=DEFAULT_PAYLOAD_MASS_G)
    parser.add_argument(
        "--roll_angles_deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROLL_ANGLES_DEG),
    )
    parser.add_argument(
        "--tilt_angles_deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_TILT_ANGLES_DEG),
    )
    parser.add_argument("--env_spacing", type=float, default=2.0)
    parser.add_argument("--physics_hz", type=float, default=240.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--warmup_seconds", type=float, default=1.0)
    parser.add_argument("--tare_duration", type=float, default=0.75)
    parser.add_argument("--min_settle_seconds", type=float, default=1.0)
    parser.add_argument("--max_settle_seconds", type=float, default=10.0)
    parser.add_argument("--stable_window_seconds", type=float, default=0.5)
    parser.add_argument("--sample_duration", type=float, default=1.0)
    parser.add_argument("--linear_speed_threshold", type=float, default=0.005)
    parser.add_argument("--angular_speed_threshold", type=float, default=0.05)
    parser.add_argument("--position_tolerance", type=float, default=0.006)
    parser.add_argument("--slide_tolerance", type=float, default=0.006)
    parser.add_argument("--orientation_tolerance_deg", type=float, default=2.0)
    parser.add_argument("--drop_clearance", type=float, default=0.0005)
    parser.add_argument("--force_conversion", action="store_true")
    parser.add_argument("--no_sample_csv", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing angle-sweep artifacts in output_dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless wrench agreement and contact checks pass.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    try:
        args.angle_pairs_deg = angle_grid(args.roll_angles_deg, args.tilt_angles_deg)
    except ValueError as error:
        parser.error(str(error))
    if len(args.roll_angles_deg) < 2 or len(args.tilt_angles_deg) < 2:
        parser.error("The experiment requires at least two Roll and two Tilt angles")
    if not any(abs(value) < 1.0e-12 for value in args.roll_angles_deg):
        parser.error("--roll_angles_deg must include 0 for the center-slice plot")
    if not any(abs(value) < 1.0e-12 for value in args.tilt_angles_deg):
        parser.error("--tilt_angles_deg must include 0 for the center-slice plot")
    if args.payload_mass_g <= 0:
        parser.error("--payload_mass_g must be positive")
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
    existing = [args.output_dir / name for name in ARTIFACT_NAMES if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(
            "Refusing to overwrite existing angle-sweep artifacts; choose a new --output_dir "
            "or pass --overwrite explicitly: " + ", ".join(str(path) for path in existing)
        )
    return args


def run(args: argparse.Namespace, app, urdf_path: Path, model: dict[str, object]) -> bool:
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.scene import InteractiveScene

    from axia80_feasibility.measurement import write_summary_csv
    from axia80_feasibility.wrist_angle_measurement import (
        evaluate_wrist_angle_acceptance,
        summarize_wrist_angle_wrenches,
        write_wrist_angle_sample_csv,
        wrist_angle_metrics,
    )
    from axia80_feasibility.wrist_angle_plotting import (
        plot_wrist_angle_center_slices,
        plot_wrist_angle_error_heatmaps,
        plot_wrist_angle_heatmaps,
    )
    from axia80_feasibility.wrist_angle_scene import make_wrist_angle_scene

    angle_pairs = list(args.angle_pairs_deg)
    num_envs = len(angle_pairs)
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
        make_wrist_angle_scene(
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
    roll_joint_ids, roll_joint_names = robot.find_joints(
        str(model["roll_joint_name"]), preserve_order=True
    )
    tilt_joint_ids, tilt_joint_names = robot.find_joints(
        str(model["tilt_joint_name"]), preserve_order=True
    )
    if len(roll_joint_ids) != 1 or len(tilt_joint_ids) != 1:
        raise RuntimeError(
            f"Expected one Roll and one Tilt joint, got roll={roll_joint_names}, "
            f"tilt={tilt_joint_names}; available={robot.joint_names}"
        )
    roll_commands = torch.tensor(
        [math.radians(roll) for roll, _ in angle_pairs],
        dtype=torch.float32,
        device=sim.device,
    )
    tilt_commands = torch.tensor(
        [math.radians(tilt) for _, tilt in angle_pairs],
        dtype=torch.float32,
        device=sim.device,
    )
    joint_target[:, roll_joint_ids[0]] += roll_commands
    joint_target[:, tilt_joint_ids[0]] += tilt_commands
    robot.write_joint_state_to_sim(joint_target, torch.zeros_like(joint_target))
    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()

    tool_ids, tool_names = robot.find_bodies(str(model["tool_link_name"]), preserve_order=True)
    sensor_ids, sensor_names = robot.find_bodies(
        str(model["sensor_link_name"]), preserve_order=True
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
    payload_mass_kg = args.payload_mass_g * 1.0e-3
    masses_cpu = torch.full((num_envs, 1), payload_mass_kg, dtype=torch.float32, device="cpu")
    original_masses = payload.root_physx_view.get_masses().clone()
    mass_ratio = masses_cpu / original_masses
    inertias = payload.data.default_inertia.cpu() * mass_ratio
    payload.root_physx_view.set_masses(masses_cpu, env_ids_cpu)
    payload.root_physx_view.set_inertias(inertias, env_ids_cpu)
    applied_masses = payload.root_physx_view.get_masses().cpu()
    if not torch.allclose(applied_masses, masses_cpu, rtol=1.0e-5, atol=1.0e-7):
        raise RuntimeError(
            f"PhysX payload masses differ from commands: commanded={masses_cpu.flatten().tolist()}, "
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

    world_up = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
    world_up[:, 2] = 1.0
    tool_mass = float(model["tool_mass_kg"])
    tool_center_local = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
    tool_center_local[:, 2] = float(model["tool_center_z_in_sensor_m"])
    payload_masses_device = masses_cpu.flatten().to(device=sim.device)

    def support_force_local(masses: torch.Tensor) -> torch.Tensor:
        support_force_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=sim.device)
        support_force_w[:, 2] = masses * args.gravity
        return math_utils.quat_apply_inverse(
            robot.data.body_quat_w[:, tool_body_id], support_force_w
        )

    def expected_tool_wrench_from_pose() -> torch.Tensor:
        masses = torch.full((num_envs,), tool_mass, dtype=torch.float32, device=sim.device)
        force = support_force_local(masses)
        moment = torch.linalg.cross(tool_center_local, force, dim=-1)
        return torch.cat((force, moment), dim=-1)

    def expected_payload_wrench_from_pose() -> torch.Tensor:
        relative_w = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        relative_local = math_utils.quat_apply_inverse(
            robot.data.body_quat_w[:, tool_body_id], relative_w
        )
        force = support_force_local(payload_masses_device)
        moment = torch.linalg.cross(relative_local, force, dim=-1)
        return torch.cat((force, moment), dim=-1)

    warmup_steps = max(1, round(args.warmup_seconds / dt))
    for _ in range(warmup_steps):
        step_simulation()

    sensor_to_tool_position = torch.linalg.vector_norm(
        robot.data.body_pos_w[:, sensor_body_id] - robot.data.body_pos_w[:, tool_body_id], dim=-1
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

    tool_quat = robot.data.body_quat_w[:, tool_body_id]
    actual_up_local = math_utils.quat_apply_inverse(tool_quat, world_up)
    nominal_up_local = torch.stack(
        (
            torch.cos(roll_commands) * torch.cos(tilt_commands),
            -torch.sin(roll_commands) * torch.cos(tilt_commands),
            -torch.sin(tilt_commands),
        ),
        dim=-1,
    )
    initial_orientation_error = torch.linalg.vector_norm(
        actual_up_local - nominal_up_local, dim=-1
    )
    if float(initial_orientation_error.max()) > 0.03 or float(actual_up_local[:, 0].min()) <= 0.0:
        raise RuntimeError(
            "The wrist angle grid does not match the commanded Roll/Tilt convention: "
            f"max up-vector error={float(initial_orientation_error.max()):.6f}, "
            f"min upward normal={float(actual_up_local[:, 0].min()):.6f}"
        )

    first_sensor = robot.data.body_pos_w[0, sensor_body_id].detach().cpu().tolist()
    sim.set_camera_view(
        eye=[first_sensor[0] + 1.0, first_sensor[1] + 0.8, first_sensor[2] + 0.6],
        target=[first_sensor[0], first_sensor[1] - 0.12, first_sensor[2]],
    )
    nominal_mu_required = max(
        math.sqrt(
            math.sin(math.radians(roll)) ** 2 * math.cos(math.radians(tilt)) ** 2
            + math.sin(math.radians(tilt)) ** 2
        )
        / (math.cos(math.radians(roll)) * math.cos(math.radians(tilt)))
        for roll, tilt in angle_pairs
    )
    nominal_friction_ratio = float(model["static_friction"]) / nominal_mu_required
    print(
        f"[INFO]: {num_envs} environments, fixed payload={args.payload_mass_g} g, "
        f"Roll={list(args.roll_angles_deg)}, Tilt={list(args.tilt_angles_deg)}"
    )
    print(
        f"[INFO]: Roll=delta wrist_3, Tilt=delta wrist_2; high friction "
        f"static/dynamic={float(model['static_friction']):.2f}/{float(model['dynamic_friction']):.2f}"
    )
    print(
        f"[INFO]: nominal maximum required friction={nominal_mu_required:.6f}, "
        f"static safety ratio={nominal_friction_ratio:.2f}x"
    )

    tare_steps = max(2, round(args.tare_duration / dt))
    tare_samples_device: list[torch.Tensor] = []
    expected_tare_samples_device: list[torch.Tensor] = []
    for _ in range(tare_steps):
        step_simulation()
        tare_samples_device.append(reaction_wrench_sensor_frame().clone())
        expected_tare_samples_device.append(expected_tool_wrench_from_pose())
    tare_samples_tensor = torch.stack(tare_samples_device)
    expected_tare_samples_tensor = torch.stack(expected_tare_samples_device)
    if not bool(torch.isfinite(tare_samples_tensor).all()):
        raise RuntimeError("Non-finite wrench detected during the tare window")
    expected_tare_mean_device = expected_tare_samples_tensor.mean(dim=0)

    payload_center_local = torch.tensor(
        model["payload_center_in_sensor_m"], dtype=torch.float32, device=sim.device
    ).repeat(num_envs, 1)
    drop_center_local = payload_center_local.clone()
    drop_center_local[:, 0] += args.drop_clearance
    payload_pose = payload.data.root_pose_w.clone()
    payload_pose[:, :3] = robot.data.body_pos_w[:, tool_body_id] + math_utils.quat_apply(
        robot.data.body_quat_w[:, tool_body_id], drop_center_local
    )
    payload_pose[:, 3:7] = robot.data.body_quat_w[:, tool_body_id]
    payload_velocity = torch.zeros((num_envs, 6), dtype=torch.float32, device=sim.device)
    gravity_enabled = torch.zeros((num_envs, 1), dtype=torch.uint8, device="cpu")
    payload.root_physx_view.set_disable_gravities(gravity_enabled, env_ids_cpu)
    payload.write_root_pose_to_sim(payload_pose)
    payload.write_root_velocity_to_sim(payload_velocity)
    payload.root_physx_view.wake_up(env_ids_cpu)
    if bool(torch.any(payload.root_physx_view.get_disable_gravities().cpu() != 0)):
        raise RuntimeError("Payload gravity did not re-enable in every environment")

    size_x, size_y, size_z = (float(value) for value in model["tool_size_sensor_frame_m"])
    payload_half = 0.5 * float(model["payload_size_m"])
    tool_start_z = float(model["tool_start_z_in_sensor_m"])
    tool_end_z = tool_start_z + size_z

    def kinematics() -> tuple[torch.Tensor, ...]:
        relative_w = payload.data.root_pos_w - robot.data.body_pos_w[:, tool_body_id]
        current_quat = robot.data.body_quat_w[:, tool_body_id]
        relative_local = math_utils.quat_apply_inverse(current_quat, relative_w)
        delta_local = relative_local - payload_center_local
        position_error = torch.linalg.vector_norm(delta_local, dim=-1)
        tangent_displacement = torch.linalg.vector_norm(delta_local[:, 1:3], dim=-1)
        width_margin = 0.5 * size_y - payload_half - torch.abs(relative_local[:, 1])
        lower_length_margin = relative_local[:, 2] - payload_half - tool_start_z
        upper_length_margin = tool_end_z - payload_half - relative_local[:, 2]
        length_margin = torch.minimum(lower_length_margin, upper_length_margin)
        payload_quat = payload.data.root_pose_w[:, 3:7]
        quat_dot = torch.abs(torch.sum(payload_quat * current_quat, dim=-1))
        relative_angle_deg = torch.rad2deg(
            2.0 * torch.acos(torch.clamp(quat_dot, min=0.0, max=1.0))
        )
        tool_velocity = robot.data.body_link_vel_w[:, tool_body_id]
        tool_point_linear_velocity = tool_velocity[:, :3] + torch.linalg.cross(
            tool_velocity[:, 3:], relative_w, dim=-1
        )
        relative_linear_velocity = payload.data.root_com_lin_vel_w - tool_point_linear_velocity
        relative_angular_velocity = payload.data.root_com_ang_vel_w - tool_velocity[:, 3:]
        speed = torch.linalg.vector_norm(relative_linear_velocity, dim=-1)
        angular_speed = torch.linalg.vector_norm(relative_angular_velocity, dim=-1)
        up_local = math_utils.quat_apply_inverse(current_quat, world_up)
        actual_roll = torch.rad2deg(torch.atan2(-up_local[:, 1], up_local[:, 0]))
        actual_tilt = torch.rad2deg(
            torch.atan2(-up_local[:, 2], torch.linalg.vector_norm(up_local[:, :2], dim=-1))
        )
        inclination = torch.rad2deg(
            torch.acos(torch.clamp(up_local[:, 0], min=-1.0, max=1.0))
        )
        required_mu = torch.linalg.vector_norm(up_local[:, 1:3], dim=-1) / torch.clamp(
            up_local[:, 0], min=1.0e-6
        )
        return (
            position_error,
            tangent_displacement,
            width_margin,
            length_margin,
            relative_angle_deg,
            speed,
            angular_speed,
            actual_roll,
            actual_tilt,
            inclination,
            required_mu,
            up_local[:, 0],
        )

    minimum_steps = max(1, round(args.min_settle_seconds / dt))
    maximum_steps = max(minimum_steps, round(args.max_settle_seconds / dt))
    required_stable_steps = max(1, round(args.stable_window_seconds / dt))
    consecutive_stable = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
    settled_mask = torch.zeros(num_envs, dtype=torch.bool, device=sim.device)
    settle_steps = maximum_steps
    for settle_step in range(maximum_steps):
        step_simulation()
        values = kinematics()
        position_error, tangent_displacement = values[0], values[1]
        width_margin, length_margin, relative_angle_deg = values[2], values[3], values[4]
        speed, angular_speed, upward_normal = values[5], values[6], values[11]
        stable_now = (
            (speed <= args.linear_speed_threshold)
            & (angular_speed <= args.angular_speed_threshold)
            & (position_error <= args.position_tolerance)
            & (tangent_displacement <= args.slide_tolerance)
            & (relative_angle_deg <= args.orientation_tolerance_deg)
            & (width_margin > 0.0)
            & (length_margin > 0.0)
            & (upward_normal > 0.0)
        )
        if settle_step + 1 < minimum_steps:
            stable_now.zero_()
        consecutive_stable = torch.where(
            stable_now, consecutive_stable + 1, torch.zeros_like(consecutive_stable)
        )
        settled_mask = consecutive_stable >= required_stable_steps
        if settle_step % max(1, round(args.physics_hz)) == 0:
            print(
                f"[INFO]: settling t={(settle_step + 1) * dt:.2f}s, "
                f"stable={int(settled_mask.sum())}/{num_envs}, "
                f"max_speed={float(speed.max()):.6f} m/s, "
                f"max_tangent_shift={float(tangent_displacement.max()):.6f} m"
            )
        if bool(settled_mask.all()):
            settle_steps = settle_step + 1
            break
    if not bool(settled_mask.all()):
        missing = torch.nonzero(~settled_mask, as_tuple=False).flatten().cpu().tolist()
        raise RuntimeError(
            "Stability/no-slide timeout: refusing to record measurements; "
            f"unsettled environment ids={missing}"
        )
    print(f"[INFO]: all angle-grid payloads stable after {settle_steps * dt:.3f} s")

    sample_steps = max(2, round(args.sample_duration / dt))
    samples_device: list[torch.Tensor] = []
    expected_samples_device: list[torch.Tensor] = []
    sample_stable_mask = torch.ones(num_envs, dtype=torch.bool, device=sim.device)
    sample_max_speed = torch.zeros(num_envs, dtype=torch.float32, device=sim.device)
    sample_max_angular_speed = torch.zeros_like(sample_max_speed)
    sample_max_position_error = torch.zeros_like(sample_max_speed)
    sample_max_tangent = torch.zeros_like(sample_max_speed)
    sample_max_relative_angle = torch.zeros_like(sample_max_speed)
    sample_min_width_margin = torch.full_like(sample_max_speed, math.inf)
    sample_min_length_margin = torch.full_like(sample_max_speed, math.inf)
    sample_roll_sum = torch.zeros_like(sample_max_speed)
    sample_tilt_sum = torch.zeros_like(sample_max_speed)
    sample_max_inclination = torch.zeros_like(sample_max_speed)
    sample_max_required_mu = torch.zeros_like(sample_max_speed)
    for _ in range(sample_steps):
        step_simulation()
        wrench = reaction_wrench_sensor_frame().clone()
        if not bool(torch.isfinite(wrench).all()):
            raise RuntimeError("Non-finite wrench detected during the sample window")
        samples_device.append(wrench)
        expected_corrected = (
            expected_payload_wrench_from_pose()
            + expected_tool_wrench_from_pose()
            - expected_tare_mean_device
        )
        expected_samples_device.append(expected_corrected)
        values = kinematics()
        (
            position_error,
            tangent_displacement,
            width_margin,
            length_margin,
            relative_angle_deg,
            speed,
            angular_speed,
            actual_roll,
            actual_tilt,
            inclination,
            required_mu,
            upward_normal,
        ) = values
        sample_max_speed = torch.maximum(sample_max_speed, speed)
        sample_max_angular_speed = torch.maximum(sample_max_angular_speed, angular_speed)
        sample_max_position_error = torch.maximum(sample_max_position_error, position_error)
        sample_max_tangent = torch.maximum(sample_max_tangent, tangent_displacement)
        sample_max_relative_angle = torch.maximum(sample_max_relative_angle, relative_angle_deg)
        sample_min_width_margin = torch.minimum(sample_min_width_margin, width_margin)
        sample_min_length_margin = torch.minimum(sample_min_length_margin, length_margin)
        sample_roll_sum += actual_roll
        sample_tilt_sum += actual_tilt
        sample_max_inclination = torch.maximum(sample_max_inclination, inclination)
        sample_max_required_mu = torch.maximum(sample_max_required_mu, required_mu)
        sample_stable_mask &= (
            (speed <= args.linear_speed_threshold)
            & (angular_speed <= args.angular_speed_threshold)
            & (position_error <= args.position_tolerance)
            & (tangent_displacement <= args.slide_tolerance)
            & (relative_angle_deg <= args.orientation_tolerance_deg)
            & (width_margin > 0.0)
            & (length_margin > 0.0)
            & (upward_normal > 0.0)
        )
    samples_tensor = torch.stack(samples_device)
    expected_samples_tensor = torch.stack(expected_samples_device)
    if not bool(sample_stable_mask.all()):
        missing = torch.nonzero(~sample_stable_mask, as_tuple=False).flatten().cpu().tolist()
        raise RuntimeError(
            "Payload slid, tipped, or lost stability during measurement; "
            f"refusing to write CSV data; environment ids={missing}"
        )

    settled_mask &= sample_stable_mask
    samples = samples_tensor.detach().cpu().numpy()
    expected_samples = expected_samples_tensor.detach().cpu().numpy()
    tare_samples = tare_samples_tensor.detach().cpu().numpy()
    expected_tare_samples = expected_tare_samples_tensor.detach().cpu().numpy()
    rows = summarize_wrist_angle_wrenches(
        angle_pairs,
        args.payload_mass_g,
        samples,
        tare_samples,
        expected_samples,
        expected_tare_samples,
        gravity_m_s2=args.gravity,
        bending_lever_arm_m=float(model["payload_bending_lever_arm_m"]),
        payload_normal_offset_m=float(model["payload_center_in_sensor_m"][0]),
        settled=settled_mask.detach().cpu().tolist(),
        settle_time_s=settle_steps * dt,
        sample_mean_actual_roll_deg=(sample_roll_sum / sample_steps).detach().cpu().tolist(),
        sample_mean_actual_tilt_deg=(sample_tilt_sum / sample_steps).detach().cpu().tolist(),
        sample_max_combined_inclination_deg=sample_max_inclination.detach().cpu().tolist(),
        sample_max_required_static_friction=sample_max_required_mu.detach().cpu().tolist(),
        sample_max_relative_speed_m_s=sample_max_speed.detach().cpu().tolist(),
        sample_max_relative_angular_speed_rad_s=sample_max_angular_speed.detach().cpu().tolist(),
        sample_max_position_error_m=sample_max_position_error.detach().cpu().tolist(),
        sample_max_tangent_displacement_m=sample_max_tangent.detach().cpu().tolist(),
        sample_max_payload_tool_angle_deg=sample_max_relative_angle.detach().cpu().tolist(),
        sample_min_width_edge_margin_m=sample_min_width_margin.detach().cpu().tolist(),
        sample_min_length_edge_margin_m=sample_min_length_margin.detach().cpu().tolist(),
    )
    metrics = wrist_angle_metrics(rows)
    passed, failures = evaluate_wrist_angle_acceptance(
        metrics,
        position_tolerance_m=min(args.position_tolerance, args.slide_tolerance),
        orientation_tolerance_deg=args.orientation_tolerance_deg,
        static_friction=float(model["static_friction"]),
    )

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "axia80_wrist_angle_summary.csv"
    samples_path = output_dir / "axia80_wrist_angle_samples.csv"
    response_path = output_dir / "axia80_wrist_angle_response_heatmaps.png"
    error_path = output_dir / "axia80_wrist_angle_error_heatmaps.png"
    slices_path = output_dir / "axia80_wrist_angle_center_slices.png"
    metadata_path = output_dir / "axia80_wrist_angle_metadata.json"
    run_log_path = output_dir / "axia80_wrist_angle_run.log"
    write_summary_csv(summary_path, rows)
    if not args.no_sample_csv:
        write_wrist_angle_sample_csv(
            samples_path,
            angle_pairs,
            args.payload_mass_g,
            samples,
            np.mean(tare_samples, axis=0),
            expected_samples,
            dt_s=dt,
        )
    plot_wrist_angle_heatmaps(summary_path, response_path)
    plot_wrist_angle_error_heatmaps(summary_path, error_path)
    plot_wrist_angle_center_slices(summary_path, slices_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary_csv": str(summary_path),
        "sample_csv": None if args.no_sample_csv else str(samples_path),
        "response_heatmaps_png": str(response_path),
        "error_heatmaps_png": str(error_path),
        "center_slices_png": str(slices_path),
        "run_log": str(run_log_path),
    }
    metadata = {
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
        "metrics": metrics,
        "payload_mass_g": args.payload_mass_g,
        "roll_angles_deg": list(args.roll_angles_deg),
        "tilt_angles_deg": list(args.tilt_angles_deg),
        "angle_pairs_deg": angle_pairs,
        "num_envs": num_envs,
        "simulation_dt_s": dt,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": model,
        "wrench_convention": {
            "order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "frame": "Axia80 center coincident with the tool body frame",
            "roll": "wrist_3 offset about tool +Z",
            "tilt": "wrist_2 offset from the horizontal cantilever pose",
            "reported_primary_columns": "per-environment tool-only tare corrected",
            "expected_columns": (
                "sampled actual-pose payload wrench plus tool pose change since tare"
            ),
        },
        "contact_checks": {
            "static_friction": float(model["static_friction"]),
            "dynamic_friction": float(model["dynamic_friction"]),
            "nominal_max_required_static_friction": nominal_mu_required,
            "nominal_friction_safety_ratio": nominal_friction_ratio,
            "slide_tolerance_m": args.slide_tolerance,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
        },
        "artifacts": artifacts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    run_log_lines = [
        f"timestamp={datetime.now().astimezone().isoformat()}",
        "experiment=UR5e-Axia80 fixed-mass wrist Roll x Tilt angle sweep",
        f"verdict={'PASS' if passed else 'FAIL'}",
        f"num_envs={num_envs}",
        f"payload_mass_g={args.payload_mass_g}",
        "roll_angles_deg=" + json.dumps(list(args.roll_angles_deg)),
        "tilt_angles_deg=" + json.dumps(list(args.tilt_angles_deg)),
        f"settle_time_s={settle_steps * dt:.9f}",
        f"tool_width_m={size_y:.9f}",
        f"static_friction={float(model['static_friction']):.9f}",
        f"dynamic_friction={float(model['dynamic_friction']):.9f}",
        f"nominal_max_required_static_friction={nominal_mu_required:.9f}",
        f"nominal_friction_safety_ratio={nominal_friction_ratio:.9f}",
        "metrics=" + json.dumps(metrics, sort_keys=True),
        "failures=" + json.dumps(failures),
        "artifacts=" + json.dumps(artifacts, sort_keys=True),
    ]
    run_log_path.write_text("\n".join(run_log_lines) + "\n")

    print(f"[RESULT]: {'PASS' if passed else 'FAIL'}")
    print("[RESULT]: " + json.dumps(metrics, sort_keys=True))
    for failure in failures:
        print(f"[RESULT]: check failed: {failure}")
    print(f"[RESULT]: summary CSV:      {summary_path}")
    if not args.no_sample_csv:
        print(f"[RESULT]: sample CSV:       {samples_path}")
    print(f"[RESULT]: response heatmap: {response_path}")
    print(f"[RESULT]: error heatmap:    {error_path}")
    print(f"[RESULT]: center slices:    {slices_path}")
    print(f"[RESULT]: metadata:         {metadata_path}")
    print(f"[RESULT]: run log:          {run_log_path}")
    return passed


def main() -> None:
    args = arguments()
    root = (args.repo_root or repo_root()).resolve()
    urdf_path, model = build_wrist_angle_model(
        root, PACKAGE_ROOT / "generated/wrist_roll_tilt_sweep"
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
