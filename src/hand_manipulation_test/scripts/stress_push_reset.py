#!/usr/bin/env python3
"""Stress small asynchronous resets across the full training grid.

Optionally load a saved run's original NumPy solver, capture its first failure,
and retry the identical Cube, hand synergy and palm target with the active fix.
No simulation step, command resampling or collision bypass occurs in replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--baseline-source", type=Path, help="Saved source/hand_manipulation_test/mdp/push_events.py.")
parser.add_argument("--fixed-random-resets", action="store_true", help="Use baseline only for the deterministic precision fixture, then stress the corrected solver.")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = launcher.app


def main():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import hand_manipulation_test  # noqa: F401
    from hand_manipulation_test.action_math import inspire_synergy_to_joint_positions

    if args.samples < 1 or not 1 <= args.batch_size <= min(32, args.num_envs):
        raise ValueError("samples must be positive; batch-size must be in [1,min(32,num_envs)]")
    cfg = load_cfg_from_registry("Isaac-Hand-Manipulation-Push-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    env = gym.make("Isaac-Hand-Manipulation-Push-v0", cfg=cfg).unwrapped
    results = {"num_envs": args.num_envs, "batch_size": args.batch_size, "seed": args.seed,
               "partial_reset_rows": 0, "boundary_reset_rows": 0, "baseline_failure": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        env.reset(seed=args.seed)
        robot = env.scene["robot"]
        cube = env.scene["target_object"]
        safe = env.event_manager.get_term_cfg("safe_hand").func
        command = env.command_manager.get_term("target_position")
        all_ids = torch.arange(env.num_envs, device=env.device)
        watched = min(1927, env.num_envs - 1)
        fixed_solver = safe._solve_seed_numpy
        baseline_solver = None
        if args.baseline_source:
            spec = importlib.util.spec_from_file_location(
                "hand_manipulation_test.mdp._reset_baseline", args.baseline_source.resolve())
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            baseline_solver = MethodType(module.PushSafePoseReset._solve_seed_numpy, safe)

        def certify(ids):
            assert bool(safe.check_final(ids).all()), "Authoritative real-FK/4mm OBB gate failed"
            assert bool((safe.position_error_m[ids] <= cfg.task.reset_position_tolerance_m).all())
            assert bool((safe.orientation_error_rad[ids] <= cfg.task.reset_orientation_tolerance_rad).all())
            assert bool((safe.ik_iterations[ids] <= safe.attempts[ids] * cfg.task.reset_max_iterations).all())
            assert bool((safe.accepted_seed_index[ids] + 1 == safe.attempts[ids]).all())
            torch.testing.assert_close(robot.data.joint_vel[ids], torch.zeros_like(robot.data.joint_vel[ids]), atol=1e-6, rtol=0.)
            torch.testing.assert_close(cube.data.root_pos_w[ids], command.target_pos_w[ids], atol=2e-6, rtol=0.)

        def replay(ids, solver):
            """Retry the existing fixed target and hand, using only bounded seeds."""
            solved = torch.zeros(len(ids), dtype=torch.bool, device=env.device)
            accepted_q = robot.data.default_joint_pos[ids].clone()
            hands = inspire_synergy_to_joint_positions(safe.initial_hand_synergy[ids])
            diagnostics = []
            for index, offset in enumerate(cfg.task.reset_joint_seed_offsets):
                rows = torch.where(~solved)[0]
                if not len(rows):
                    break
                pending = ids[rows]
                q = robot.data.default_joint_pos[pending].clone()
                q[:, safe.arm_joint_ids] += q.new_tensor(offset)
                q[:, safe.arm_joint_ids] = safe._bounded_arm(q[:, safe.arm_joint_ids], pending)
                q[:, safe.hand_joint_ids] = hands[rows]
                safe._write_reset_joint_state(q, torch.zeros_like(q), pending)
                reached = solver(pending)
                candidates = pending[reached]
                accepted = safe._check_pose_and_limits(candidates) & safe._collision_free(candidates) if len(candidates) else reached[:0]
                diagnostic = {"seed_index": index, "env_ids": pending.tolist(),
                              "position_error_m": safe.position_error_m[pending].tolist(),
                              "orientation_error_rad": safe.orientation_error_rad[pending].tolist(),
                              "converged": reached.tolist(), "candidate_ids": candidates.tolist(),
                              "accepted": accepted.tolist()}
                if len(candidates):
                    diagnostic["table_overlap_bodies"] = [
                        [robot.body_names[j] for j in torch.where(mask)[0].tolist()]
                        for mask in safe.last_table_overlap[candidates]]
                diagnostics.append(diagnostic)
                accepted_rows = rows[reached][accepted]
                accepted_q[accepted_rows] = robot.data.joint_pos[ids[accepted_rows]]
                solved[accepted_rows] = True
            if bool(solved.all()):
                safe._write_reset_joint_state(accepted_q, torch.zeros_like(accepted_q), ids)
                assert bool(safe.check_final(ids).all())
            return {"success": solved.tolist(), "seeds": diagnostics}

        if baseline_solver:
            import numpy as np
            ids = all_ids[watched:watched + 1]
            saved_q = robot.data.joint_pos[ids].clone()
            position, _ = safe._control_pose(ids)
            stored_position = position[0].cpu().numpy().astype(np.float64)
            cached_position, _ = safe._numpy_arm_kinematics.pose(
                saved_q[:, safe.arm_joint_ids].cpu().numpy().astype(np.float64),
                robot.data.root_link_pose_w[ids].cpu().numpy().astype(np.float64))
            rng = np.random.default_rng(123)
            directions = rng.normal(size=(32768, 3))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            targets = (stored_position + .003 * directions).astype(np.float32).astype(np.float64)
            cached_error = np.linalg.norm(targets - cached_position[0], axis=-1)
            stored_error = np.linalg.norm(targets - stored_position, axis=-1)
            candidates = np.flatnonzero((cached_error < .003) & (stored_error > .003))
            assert len(candidates), "Actual large-origin FK did not expose a float32 boundary disagreement"
            best = candidates[np.argmax(np.minimum(.003 - cached_error[candidates], stored_error[candidates] - .003))]
            safe.desired_c_pos_w[ids] = torch.as_tensor(targets[best], dtype=position.dtype, device=env.device)
            safe.ik_iterations[ids] = 0
            baseline_converged = baseline_solver(ids)
            legacy_error = float(safe.position_error_m[watched])
            assert not bool(baseline_converged.any()), "Baseline must reject the stored candidate after cached convergence"
            safe._write_reset_joint_state(saved_q, torch.zeros_like(saved_q), ids)
            safe.ik_iterations[ids] = 0
            fixed_converged = fixed_solver(ids)
            assert bool(fixed_converged.all()) and bool(safe.check_final(ids).all())
            results["precision_boundary_fixture"] = {
                "env_id": watched, "env_origin_m": env.scene.env_origins[watched].tolist(),
                "baseline_cached_error_m": float(cached_error[best]), "baseline_real_error_m": legacy_error,
                "fixed_real_error_m": float(safe.position_error_m[watched]),
                "fixed_orientation_error_rad": float(safe.orientation_error_rad[watched]),
                "final_collision_gate_passed": True,
            }
            print("REAL_PRECISION_FIXTURE_PASS " + json.dumps(results["precision_boundary_fixture"]), flush=True)
            safe._solve_seed_numpy = fixed_solver
            env.reset(env_ids=ids)
            if args.fixed_random_resets:
                baseline_solver = None

        started = time.perf_counter()
        sim_counter = env._sim_step_counter
        for sample in range(args.samples):
            # Include the reported large-origin row in every small batch.
            permutation = torch.randperm(env.num_envs, device=env.device)
            ids = torch.cat((all_ids[watched:watched + 1], permutation[permutation != watched][:args.batch_size - 1]))
            untouched = all_ids[~torch.isin(all_ids, ids)]
            old_q = robot.data.joint_pos[untouched].clone()
            old_goal = command.goal_pos_w[untouched].clone()
            safe._solve_seed_numpy = baseline_solver or fixed_solver
            try:
                env.reset(env_ids=ids)
            except RuntimeError as error:
                if baseline_solver is None or "Palm-facing reset failed" not in str(error):
                    raise
                bad = ids[safe.accepted_seed_index[ids] < 0]
                fixture = {"env_ids": bad.cpu(), "env_origins": env.scene.env_origins[bad].cpu(),
                           "root_pose_w": robot.data.root_link_pose_w[bad].cpu(),
                           "cube_position_w": cube.data.root_pos_w[bad].cpu(),
                           "desired_c_pos_w": safe.desired_c_pos_w[bad].cpu(),
                           "desired_c_quat_w": safe.desired_c_quat_w[bad].cpu(),
                           "direction_w": safe.direction_w[bad].cpu(),
                           "hand_synergy": safe.initial_hand_synergy[bad].cpu()}
                torch.save(fixture, args.output.with_suffix(".fixture.pt"))
                # Preserve the failed sampled state while comparing solvers.
                safe.ik_iterations[bad] = 0
                original_replay = replay(bad, baseline_solver)
                safe.ik_iterations[bad] = 0
                fixed_replay = replay(bad, fixed_solver)
                results["baseline_failure"] = {"sample": sample, "error": str(error),
                                               "original_replay": original_replay, "fixed_replay": fixed_replay}
                assert all(fixed_replay["success"]), "Active solver could not certify the identical failed fixture"
                print("BASELINE_FAILURE_REPAIRED " + json.dumps(results["baseline_failure"]), flush=True)
                # Recover all rows through Manager reset and continue stress
                # exclusively with the corrected backend.
                safe._solve_seed_numpy = fixed_solver
                baseline_solver = None
                env.reset(env_ids=ids)
            certify(ids)
            assert env._sim_step_counter == sim_counter, "IK reset stepped physics"
            torch.testing.assert_close(robot.data.joint_pos[untouched], old_q, atol=0., rtol=0.)
            torch.testing.assert_close(command.goal_pos_w[untouched], old_goal, atol=0., rtol=0.)
            results["partial_reset_rows"] += len(ids)
            if (sample + 1) % 100 == 0:
                print(f"RESET_STRESS {sample + 1}/{args.samples}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
        safe._solve_seed_numpy = fixed_solver
        # Actual grid extremes, both angle/length boundaries and Cube corners.
        boundary_ids = all_ids[torch.tensor(sorted(set((0, watched, env.num_envs // 2, env.num_envs - 1))), device=env.device)]
        from itertools import product
        for x, y, degrees, distance in product(
            (cfg.task.object_xy_range_low[0], cfg.task.object_xy_range_high[0]),
            (cfg.task.object_xy_range_low[1], cfg.task.object_xy_range_high[1]),
            (-10., 0., 10., 170., 180., 190.), cfg.task.command_distance_range):
            command.set_reset_specs(boundary_ids, (x, y, cfg.task.cube_center_height_m), math.radians(degrees), distance)
            env.reset(env_ids=boundary_ids)
            certify(boundary_ids)
            results["boundary_reset_rows"] += len(boundary_ids)
        # Exercise the automatic selection boundary back to PhysX.
        if env.num_envs >= 33:
            ids = torch.cat((all_ids[watched:watched + 1], all_ids[all_ids != watched][:32]))
            env.reset(env_ids=ids)
            certify(ids)
            results["physx_branch_rows"] = len(ids)
        results["elapsed_seconds"] = time.perf_counter() - started
        results["passed"] = True
        print("RESET_STRESS_PASS " + json.dumps(results), flush=True)
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(status)
        simulation_app.close()
    raise SystemExit(status)
