"""Offline FK/OBB exploration of reset and side-contact poses, no physics rollout."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/hand_manipulation_test"))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("contact_pose_grid.json"))
parser.add_argument("--relax-wrist", action="store_true")
parser.add_argument("--contact-only", action="store_true")
parser.add_argument("--warm-start", action="store_true")
parser.add_argument("--table-x", type=float, default=.36)
parser.add_argument("--cube-x", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
app = launcher.app

def main():
    import gymnasium as gym
    import torch
    import hand_manipulation_test
    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from hand_manipulation_test.push_math import push_palm_rotation
    cfg = load_cfg_from_registry("Isaac-Hand-Manipulation-Push-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.task.object_xy_offset_low = (-.065, -.06)
    cfg.task.object_xy_offset_high = (.08, .06)
    cfg.task.table_size = (args.table_x, 1., .04)
    cfg.scene.table.spawn.size = cfg.task.table_size
    if args.relax_wrist:
        cfg.task.reset_wrist_3_range_rad = (-math.pi, math.pi)
    env = gym.make("Isaac-Hand-Manipulation-Push-v0", cfg=cfg).unwrapped
    report = {"table_center": cfg.task.table_center, "cases": []}
    try:
        safe = env.event_manager.get_term_cfg("safe_hand").func
        command = env.command_manager.get_term("target_position")
        ids = torch.arange(4, device=env.device)
        offsets = torch.zeros((4, 3), device=env.device)
        rolls = torch.zeros(4, device=env.device)

        def sample(env_ids):
            cube = env.scene["target_object"].data.root_pos_w[env_ids]
            direction = command.direction_w[env_ids]
            rotation = push_palm_rotation(direction)
            up = torch.zeros_like(direction)
            up[:, 2] = 1
            z = rolls[env_ids].cos()[:, None] * rotation[:, :, 2] - rolls[env_ids].sin()[:, None] * up
            x = torch.linalg.cross(direction, z, dim=-1)
            rotation = torch.stack((x, direction, z), dim=-1)
            palm = cube - offsets[env_ids, :1] * direction + offsets[env_ids, 2:] * up
            safe.desired_palm_position_w[env_ids] = palm
            safe.desired_c_pos_w[env_ids] = palm + (rotation @ (safe.c_offset_h - safe.palm_reference_h).unsqueeze(-1)).squeeze(-1)
            q = math_utils.quat_from_matrix(rotation)
            safe.desired_c_quat_w[env_ids] = math_utils.quat_unique(math_utils.quat_mul(q, safe.c_quat_h.expand(len(env_ids), -1)))
            safe.direction_w[env_ids] = direction
        safe._sample_pose = sample
        original_solve = safe._solve_seed
        warm_q = None
        def solve(env_ids, hand_targets):
            if warm_q is not None:
                q = safe.robot.data.joint_pos[env_ids].clone()
                q[:, safe.arm_joint_ids] = warm_q[env_ids][:, safe.arm_joint_ids]
                safe._write_reset_joint_state(q, torch.zeros_like(q), env_ids)
            return original_solve(env_ids, hand_targets)
        safe._solve_seed = solve
        angles = torch.tensor([0., 0., math.pi, math.pi], device=env.device)
        phases = (("contact_clearance", .038),) if args.contact_only else (("start", .14), ("contact_clearance", .038))
        for phase, backoff in phases:
            x_grid = (args.cube_x,) if args.cube_x is not None else (-.70, -.78, -.80, -.81)
            for cube_x in x_grid:
                degree_grid = (0., 25., 30., 35.) if args.warm_start else (0., 15., 25., 30., 35., 40., 45., 60.)
                for degrees in degree_grid:
                    heights = (.10,) if phase == "start" else ((.025, .04, .055) if args.warm_start else (.01, .025, .04, .055))
                    for height in heights:
                        offsets[:, 0] = backoff
                        offsets[:, 2] = height
                        rolls[:] = 0
                        rolls[2:] = math.radians(degrees)
                        positions = torch.tensor([(cube_x, -.06, cfg.task.cube_center_height_m),
                                                  (cube_x, .06, cfg.task.cube_center_height_m)] * 2, device=env.device)
                        if args.warm_start:
                            warm_q = None
                            offsets[:, 0] = .14
                            offsets[:, 2] = .10
                            command.set_reset_specs(ids, positions, angles, .30)
                            try:
                                env.reset()
                            except RuntimeError:
                                pass
                            warm_q = safe.robot.data.joint_pos.clone()
                            offsets[:, 0] = backoff
                            offsets[:, 2] = height
                        command.set_reset_specs(ids, positions, angles, .30)
                        failure = None
                        try:
                            env.reset()
                        except RuntimeError as error:
                            failure = str(error).split(": [")[0]
                        valid = safe.accepted_seed_index >= 0
                        cube_mask = safe.last_cube_overlap
                        table_mask = safe.last_table_overlap
                        case = {"phase": phase, "cube_x": cube_x, "left_roll_down_degrees": degrees,
                                "palm_height_m": height, "valid": valid.tolist(),
                                "seed": safe.accepted_seed_index.tolist(),
                                "table_overlap": [[name for name, hit in zip(safe.robot.body_names, row) if hit]
                                                   for row in table_mask.cpu().tolist()],
                                "cube_overlap": [[name for name, hit in zip(safe.robot.body_names, row) if hit]
                                                  for row in cube_mask.cpu().tolist()],
                                "position_error_m": safe.position_error_m.tolist(), "failure": failure}
                        case["orientation_error_rad"] = safe.orientation_error_rad.tolist()
                        case["arm_q"] = safe.robot.data.joint_pos[:, safe.arm_joint_ids].tolist()
                        case["seed_diagnostics"] = [safe.seed_diagnostics(i) for i in range(4)]
                        if warm_q is not None:
                            case["warm_start_arm_q"] = warm_q[:, safe.arm_joint_ids].tolist()
                        report["cases"].append(case)
                        if bool(valid.all()) or (phase == "start" and degrees in (30., 40.)):
                            print("POSE_GRID", json.dumps(case), flush=True)
                        args.output.write_text(json.dumps(report, indent=2))
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
