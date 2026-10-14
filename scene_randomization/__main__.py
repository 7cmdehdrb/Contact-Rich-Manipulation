"""Run with python -m scene_randomization --count 10000 --output outputs/scene_randomization."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .generator import Config, GenerationError, generate_parallel
from .validation import distribution_report, validate_scenes


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast simulator-free push scene generation and geometry audit (cm)")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", type=Path, help="JSON object of Config overrides")
    parser.add_argument("--output", type=Path, default=Path("outputs/scene_randomization"))
    parser.add_argument("--plots", type=int, default=12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-save-scenes", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    c = Config(**overrides)
    c.check()
    if args.plots < 1:
        parser.error("--plots must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        batch = generate_parallel(args.count, c, args.seed, args.workers)
    except GenerationError as exc:
        failure = {"error": str(exc), "indices": exc.indices.tolist(),
                   "angle": exc.angle.tolist(), "distance": exc.distance.tolist(), "reasons": exc.reasons,
                   "seed": args.seed, "config": asdict(c)}
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2))
        print(str(exc))
        return 2
    elapsed = time.perf_counter() - started
    started = time.perf_counter()
    geometry = validate_scenes(batch)
    audit_elapsed = time.perf_counter() - started
    report = {"seed": args.seed, "workers": args.workers, "config": asdict(c),
              "generation_seconds": elapsed, "scenes_per_second": args.count / elapsed,
              "amortized_ms_per_scene": 1000 * elapsed / args.count,
              "validation_seconds": audit_elapsed,
              "attempts_mean": float(batch.attempts.mean()), "attempts_max": int(batch.attempts.max()),
              "retried_scenes": int((batch.attempts > 1).sum()),
              "geometry": geometry, "distribution": distribution_report(batch)}
    # Include visualization/serialization separately from measured generation.
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if not args.no_save_scenes:
        batch.save(str(args.output / "scenes.npz"))
    if not args.no_plots:
        os.environ.setdefault("MPLCONFIGDIR", str(args.output.resolve() / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        from .plotting import draw_scene, save_distribution, save_gallery, save_zone_explanation
        import matplotlib.pyplot as plt
        save_gallery(batch, args.output / "gallery.png", args.plots)
        save_distribution(batch, args.output / "distribution.png")
        save_zone_explanation(args.output / "zone_logic.png")
        fig, ax = plt.subplots(figsize=(12, 5))
        draw_scene(ax, batch, 0, resolution=0.25)
        ax.set(xlabel="x (cm)", ylabel="y (cm)")
        fig.tight_layout()
        fig.savefig(args.output / "scene_0000.png", dpi=180)
        plt.close(fig)
    print(f"{args.count:,} scenes: {elapsed:.3f} s, {args.count / elapsed:,.0f} scenes/s, "
          f"{1000 * elapsed / args.count:.3f} ms/scene (amortized)")
    print(f"Geometry: {'PASS' if geometry['valid'] else 'FAIL'}; "
          f"retry mean/max: {batch.attempts.mean():.2f}/{batch.attempts.max()}")
    dist = report["distribution"]
    print(f"Joint-bin counts: {dist['joint_count_min']}..{dist['joint_count_max']}")
    print(f"Artifacts: {args.output.resolve()}")
    return 0 if geometry["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
