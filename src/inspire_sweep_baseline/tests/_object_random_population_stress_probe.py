"""Repeat real ObjectRandom resets with fixed counts and unchanged rejection rules.

Run one fresh simulator per asset seed. A placement failure ends that seed's
trial immediately and is recorded before simulator teardown.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument("--population", type=int, choices=(6, 7), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--resets", type=int, default=20)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
report = dict(population=args.population, asset_seed=args.seed, num_envs=args.num_envs,
              planned_batches=args.resets, iteration=14000, padding_fraction=.15,
              status="starting", passed_batches=0, passed_layouts=0, batches=[])
args.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


save()
app = AppLauncher(args).app
env = None
started = time.monotonic()
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.mdp.common import contact_sweep_clutter_fusion_holding_osc_object_random_placement as placement
    from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat

    task = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v2"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    cfg.log_dir = str(args.output.parent)
    cfg.curriculum.obstacle_population.params.update(final_gray=args.population, final_white=args.population)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    raw.set_learning_iteration(14000)
    spawn = raw.event_manager.get_term_cfg("object_spawn").params
    assert spawn["gray_obstacles"] == spawn["white_obstacles"] == args.population
    assert spawn["gray_red_padding_fraction"] == .15
    original_sample = placement.sample_layout
    original_valid = placement.valid_layout
    attempt_counts = dict(obb_checks=0, obb_rejections=0)

    def tracked_valid(*a, **kw):
        attempt_counts["obb_checks"] += 1
        valid = original_valid(*a, **kw)
        attempt_counts["obb_rejections"] += int(not valid)
        return valid

    def tracked_sample(env_arg, ids, generator, target, goal, gray, white):
        report["last_sample"] = dict(env_ids=ids.cpu().tolist(), target=target.cpu().tolist(),
                                   goal=goal.cpu().tolist(), gray=gray, white=white)
        return original_sample(env_arg, ids, generator, target, goal, gray, white)

    placement.sample_layout, placement.valid_layout = tracked_sample, tracked_valid
    report["status"] = "running"
    for batch in range(args.resets):
        reset_seed = args.seed * 10000 + batch
        report["current_batch"], report["reset_seed"] = batch, reset_seed
        attempt_counts.update(obb_checks=0, obb_rejections=0)
        save()
        begin = time.monotonic()
        try:
            obs, _ = env.reset(seed=reset_seed)
        except RuntimeError as exc:
            if "No non-overlapping enlarged layout" not in str(exc):
                raise
            report.update(status="placement_failure", error=str(exc),
                          elapsed_seconds=time.monotonic() - started,
                          failed_batch_seconds=time.monotonic() - begin,
                          failed_attempts=dict(attempt_counts))
            save()
            print(json.dumps({key: report[key] for key in ("status", "asset_seed", "reset_seed", "error")}), flush=True)
            break
        assert all(torch.isfinite(value).all() for value in obs.values())
        expected = 1 + 2 * args.population
        assert (raw._active_object_mask.sum(-1) == expected).all()
        # Independently recheck the written state and every physical OBB.
        data = raw.scene["object_collection"].data
        shelf = raw.scene["shelf"].data
        local = quat_apply_inverse(shelf.root_quat_w[:, None].expand(-1, data.object_pos_w.shape[1], -1),
                                   data.object_pos_w - shelf.root_pos_w[:, None])
        indices = [0, *range(1, 1 + args.population),
                   *range(1 + raw.cfg.max_gray_obstacles, 1 + raw.cfg.max_gray_obstacles + args.population)]
        from isaaclab.utils.math import quat_mul, quat_conjugate
        quat = quat_mul(quat_conjugate(shelf.root_quat_w[:, None]).expand_as(data.object_quat_w), data.object_quat_w)
        yaw = euler_xyz_from_quat(quat.reshape(-1, 4))[2].reshape(quat.shape[:-1]).cpu().numpy()
        centers = local[..., :2].cpu().numpy()
        for row in range(raw.num_envs):
            assert original_valid(centers[row, indices], yaw[row, indices], raw._prop_half_extents[row, indices])
        report["passed_batches"] += 1
        report["passed_layouts"] += args.num_envs
        report["batches"].append(dict(batch=batch, reset_seed=reset_seed,
                                      seconds=time.monotonic() - begin, **attempt_counts))
        save()
        print(f"RESET_OK population={args.population} seed={args.seed} batch={batch} layouts={report['passed_layouts']}", flush=True)
    else:
        report["status"] = "passed"
    report["elapsed_seconds"] = time.monotonic() - started
    save()
    print(json.dumps({key: report[key] for key in ("population", "asset_seed", "status", "passed_layouts", "elapsed_seconds")}), flush=True)
except Exception as exc:
    report.update(status="probe_error", error=repr(exc), elapsed_seconds=time.monotonic() - started)
    save()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
