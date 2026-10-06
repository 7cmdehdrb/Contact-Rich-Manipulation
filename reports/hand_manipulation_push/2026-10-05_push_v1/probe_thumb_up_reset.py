"""GPU reset workspace grid for both-thumb-up poses; no physics rollout.

Run with the Isaac Lab interpreter and ``--headless --device cuda:0``.
The scene is created once. Each case uses the production bounded three-seed
reset and complete robot/Cube/Table OBB checks, with an exact pose fixture.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/hand_manipulation_test"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("thumb_up_reset_grid.json"))
parser.add_argument("--table-width", type=float, default=.74)
parser.add_argument("--cube-x", type=float, nargs="+", default=[-.55, -.70, -.65, -.60, -.56, -.54, -.50, -.48])
parser.add_argument("--height", type=float, nargs="+", default=[.05, .10, .03])
parser.add_argument("--backoff", type=float, nargs="+", default=[.10, .14])
parser.add_argument("--y-span", type=float, nargs="+", default=[.03, .06])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-cases", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
app = launcher.app


def save_report(report):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)


def main():
    import gymnasium as gym
    import torch
    import hand_manipulation_test  # noqa: F401 -- task registration
    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from hand_manipulation_test.geometry import control_point_pose_w
    from hand_manipulation_test.push_v1_math import push_v1_palm_rotation

    task_name = "Isaac-Hand-Manipulation-Push-v1"
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.task.table_size = (args.table_width, 1., .04)
    cfg.scene.table.spawn.size = cfg.task.table_size
    # The Table center and robot mounting remain the production configuration.
    # Wide initial bounds are reduced to a five-mm box for each exact fixture.
    centre_x = cfg.task.table_center[0]
    radius = math.sqrt(3.) * cfg.task.cube_size / 2 + cfg.task.table_path_margin_m
    safe_low_x = centre_x - args.table_width / 2 + radius
    safe_high_x = centre_x + args.table_width / 2 - radius
    initial_low_x = max(min(args.cube_x) - .005, safe_low_x)
    initial_high_x = min(max(args.cube_x) + .005, safe_high_x)
    if initial_low_x > initial_high_x:
        raise ValueError("The requested Table width cannot contain any grid Cube position")
    cfg.task.object_xy_offset_low = (initial_low_x - centre_x, -max(args.y_span) - .005)
    cfg.task.object_xy_offset_high = (initial_high_x - centre_x, max(args.y_span) + .005)
    initial_cube_x = min(max(args.cube_x[0], initial_low_x), initial_high_x)
    cfg.scene.target_object.init_state.pos = (initial_cube_x, 0., cfg.task.cube_center_height_m)
    report = {
        "task": task_name,
        "device": args.device,
        "table_center_m": cfg.task.table_center,
        "table_size_m": cfg.task.table_size,
        "cube_size_m": cfg.task.cube_size,
        "command_distance_m": .20,
        "command_angle_rad": [0., 0., math.pi, math.pi],
        "hand_open_range": cfg.task.initial_hand_open_range,
        "reset_seed_offsets": cfg.task.reset_joint_seed_offsets,
        "reset_max_iterations": cfg.task.reset_max_iterations,
        "position_tolerance_m": cfg.task.reset_position_tolerance_m,
        "orientation_tolerance_rad": cfg.task.reset_orientation_tolerance_rad,
        "clearance_margin_m": cfg.task.reset_clearance_margin_m,
        "orientation_safe_cube_radius_plus_margin_m": radius,
        "exact_pose_without_jitter": True,
        "cases": [],
    }
    env = gym.make(task_name, cfg=cfg).unwrapped
    try:
        safe = env.event_manager.get_term_cfg("safe_hand").func
        command = env.command_manager.get_term("target_position")
        hand = env.action_manager.get_term("hand_action")
        ids = torch.arange(4, device=env.device)
        offsets = torch.zeros((4, 3), device=env.device)
        angles = torch.tensor((0., 0., math.pi, math.pi), device=env.device)
        seed_q = [[] for _ in range(4)]
        original_solve = safe._solve_seed

        def solve(index, hand_targets):
            previous_iterations = safe.ik_iterations[index].clone()
            result = original_solve(index, hand_targets)
            actual = safe.robot.data.joint_pos[index]
            q_arm = actual[:, safe.arm_joint_ids].cpu().tolist()
            q_hand = actual[:, safe.hand_joint_ids].cpu().tolist()
            converged = result.cpu().tolist()
            iterations = (safe.ik_iterations[index] - previous_iterations).cpu().tolist()
            for row, env_id in enumerate(index.cpu().tolist()):
                seed_q[env_id].append({
                    "seed_index": int(safe.attempts[env_id]) - 1,
                    "arm_q_rad": q_arm[row], "hand_q_rad": q_hand[row],
                    "solver_converged": converged[row],
                    "ik_iterations": iterations[row],
                })
            return result

        def sample(index):
            cube = env.scene["target_object"].data.root_pos_w[index]
            direction = command.direction_w[index]
            rotation = push_v1_palm_rotation(direction)
            tangent = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
            up = torch.zeros_like(direction)
            up[:, 2] = 1.
            palm = cube - offsets[index, :1] * direction + offsets[index, 1:2] * tangent + offsets[index, 2:] * up
            control = palm + (rotation @ (safe.c_offset_h - safe.palm_reference_h).unsqueeze(-1)).squeeze(-1)
            local_y = control[:, 1] - env.scene.env_origins[index, 1]
            if not bool(((local_y >= env.cfg.task.reset_c_y_range[0]) & (local_y <= env.cfg.task.reset_c_y_range[1])).all()):
                raise RuntimeError("Fixture initial C is outside production reset_c_y_range")
            safe.desired_palm_position_w[index] = palm
            safe.desired_c_pos_w[index] = control
            safe.direction_w[index] = rotation[:, :, 1]
            quaternion = math_utils.quat_from_matrix(rotation)
            safe.desired_c_quat_w[index] = math_utils.quat_unique(math_utils.quat_mul(
                quaternion, safe.c_quat_h.expand(len(index), -1)
            ))

        safe._solve_seed = solve
        safe._sample_pose = sample
        grid = itertools.product(args.cube_x, args.backoff, args.height, args.y_span)
        for case_index, (cube_x, backoff, height, span) in enumerate(grid):
            if args.max_cases is not None and case_index >= args.max_cases:
                break
            env.cfg.task.object_xy_offset_low = (cube_x - centre_x - .005, -span - .005)
            env.cfg.task.object_xy_offset_high = (cube_x - centre_x + .005, span + .005)
            offsets[:, 0] = backoff
            offsets[:, 1] = 0.
            offsets[:, 2] = height
            seed_q = [[] for _ in range(4)]
            position = torch.tensor([(cube_x, -span, cfg.task.cube_center_height_m),
                                     (cube_x, span, cfg.task.cube_center_height_m)] * 2, device=env.device)
            case = {"cube_x_m": cube_x, "palm_backoff_m": backoff,
                    "palm_height_m": height, "cube_y_abs_m": span,
                    "cube_positions_local_m": position.cpu().tolist(),
                    "fixture_valid": True, "failure": None}
            try:
                command.set_reset_specs(ids, position, angles, .20)
            except ValueError as error:
                case.update(fixture_valid=False, failure=str(error))
                report["cases"].append(case)
                save_report(report)
                print("THUMB_UP_GRID_SKIP", json.dumps(case), flush=True)
                continue
            started = time.perf_counter()
            try:
                env.reset(seed=args.seed + case_index)
            except RuntimeError as error:
                case["failure"] = str(error)
            if torch.cuda.is_available() and str(env.device).startswith("cuda"):
                torch.cuda.synchronize()
            case["reset_wall_time_s"] = time.perf_counter() - started
            accepted = safe.accepted_seed_index >= 0
            certified = torch.zeros(4, dtype=torch.bool, device=env.device)
            accepted_ids = ids[accepted]
            if len(accepted_ids):
                certified[accepted_ids] = safe.check_final(accepted_ids)
            c_pos, c_quaternion = control_point_pose_w(env)
            palm_pos, hand_quaternion = safe.palm_reference_pose_w()
            thumb_up = math_utils.quat_apply(hand_quaternion, position.new_tensor((1., 0., 0.)).expand(4, -1))[:, 2]
            case.update(
                valid=accepted.cpu().tolist(), final_certified=certified.cpu().tolist(),
                seed=safe.accepted_seed_index.cpu().tolist(),
                attempts=safe.attempts.cpu().tolist(), ik_iterations=safe.ik_iterations.cpu().tolist(),
                obb_checks=safe.obb_checks.cpu().tolist(),
                position_error_m=safe.position_error_m.cpu().tolist(),
                orientation_error_rad=safe.orientation_error_rad.cpu().tolist(),
                collision_checked=safe.last_collision_checked.cpu().tolist(),
                thumb_up_cos=thumb_up.cpu().tolist(),
                actual_c_position_local_m=(c_pos - env.scene.env_origins).cpu().tolist(),
                actual_c_quaternion_wxyz=c_quaternion.cpu().tolist(),
                desired_c_position_local_m=(safe.desired_c_pos_w - env.scene.env_origins).cpu().tolist(),
                desired_c_quaternion_wxyz=safe.desired_c_quat_w.cpu().tolist(),
                actual_palm_position_local_m=(palm_pos - env.scene.env_origins).cpu().tolist(),
                desired_palm_position_local_m=(safe.desired_palm_position_w - env.scene.env_origins).cpu().tolist(),
                actual_arm_q_rad=safe.robot.data.joint_pos[:, safe.arm_joint_ids].cpu().tolist(),
                actual_hand_q_rad=safe.robot.data.joint_pos[:, safe.hand_joint_ids].cpu().tolist(),
                initial_hand_synergy=safe.initial_hand_synergy.cpu().tolist(),
                actual_hand_synergy=hand.actual_synergy.cpu().tolist(),
                table_overlap=[[name for name, hit in zip(safe.robot.body_names, row) if hit]
                               for row in safe.last_table_overlap.cpu().tolist()],
                cube_overlap=[[name for name, hit in zip(safe.robot.body_names, row) if hit]
                              for row in safe.last_cube_overlap.cpu().tolist()],
            )
            diagnostics = [safe.seed_diagnostics(i) for i in range(4)]
            for env_id, history in enumerate(diagnostics):
                by_seed = {entry["seed_index"]: entry for entry in seed_q[env_id]}
                for entry in history:
                    entry.update(by_seed.get(entry["seed_index"], {}))
            case["seed_diagnostics"] = diagnostics
            report["cases"].append(case)
            save_report(report)
            summary = {name: case[name] for name in (
                "cube_x_m", "palm_backoff_m", "palm_height_m", "cube_y_abs_m", "valid",
                "final_certified", "seed", "position_error_m", "orientation_error_rad", "table_overlap", "cube_overlap",
            )}
            print("THUMB_UP_GRID", json.dumps(summary), flush=True)
        cases = [case for case in report["cases"] if case["fixture_valid"]]
        report["summary"] = {
            "total_cases": len(report["cases"]), "valid_fixture_cases": len(cases),
            "all_four_accepted_cases": sum(all(case["valid"]) for case in cases),
            "all_four_certified_cases": sum(all(case["final_certified"]) for case in cases),
        }
        save_report(report)
        print("THUMB_UP_GRID_SUMMARY", json.dumps(report["summary"]), flush=True)
    finally:
        env.close()


status = 0
try:
    main()
except Exception:
    traceback.print_exc()
    status = 1
finally:
    import omni.kit.app
    sys.stdout.flush()
    omni.kit.app.get_app().post_quit(status)
    app.close()
raise SystemExit(status)
