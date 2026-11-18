#!/usr/bin/env python3
"""Generate OBB fixtures, infer without GT access, save reproducible evidence."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src/centroid_belief"), str(ROOT / "src/observed_obb")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/centroid_belief_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from observed_obb.observation import _footprint
from centroid_belief import Config, Posterior, SHAPES, estimate_particles as estimate
from centroid_belief.fixtures import standard_cases


def panel(ax, case, posterior, grid, metrics, *, zoom=False):
    # Horizontal=lateral Y, vertical=depth X; observer is above the plot (+X).
    density = grid["density"].T
    mesh = ax.pcolormesh(grid["y"]*100, grid["x"]*100, density / 10000,
                         cmap="Blues", shading="auto", rasterized=True)
    if density.max() > grid["threshold"]:
        ax.contour(grid["y"]*100, grid["x"]*100, density,
                   levels=[grid["threshold"]], colors=["#335588"], linewidths=1)
    for i, box in enumerate(case.scene):
        ax.add_patch(Polygon(_footprint(box)[:, ::-1]*100, fill=False,
                             edgecolor="#444444", linestyle="--", linewidth=1.1))
    for observation in case.observations:
        target = observation.observation_id == case.target_id
        ax.add_patch(Polygon(_footprint(observation.box)[:, ::-1]*100, fill=False,
                             edgecolor="#d48000" if target else "#777777", linewidth=1.5))
    ax.plot(case.truth.center[1]*100, case.truth.center[0]*100, "*", color="#111111", ms=10)
    observed = next(o.box for o in case.observations if o.observation_id == case.target_id)
    ax.plot(observed.center[1]*100, observed.center[0]*100, "x", color="#d48000", ms=7)
    ax.plot(posterior.mean[1]*100, posterior.mean[0]*100, "o", color="#c93544", ms=4)
    ax.plot(grid["map"][1]*100, grid["map"][0]*100, "+", color="#7f3b9d", ms=7)
    title = (f"{case.shape.replace('003_', '').replace('004_', '').replace('005_', '').replace('002_', '')} | {case.pattern}\n"
                 f"obs {metrics['observed_error_cm']:.2f} -> mean {metrics['mean_error_cm']:.2f} cm; "
                 f"90% {'hit' if metrics['covered90'] else 'MISS'}")
    if metrics.get("warnings"):
        title += "\nWARNING: " + metrics["warnings"].replace("_", " ")
    ax.set_title(title, fontsize=9, color="#a4262c" if metrics.get("warnings") else "black")
    ax.set_aspect("equal")
    extent = max(np.ptp(_footprint(case.truth)[:, 1])*(70 if zoom else 100), 7)
    ax.set_xlim(case.truth.center[1]*100-extent, case.truth.center[1]*100+extent)
    ax.set_ylim(case.truth.center[0]*100-9, case.truth.center[0]*100+23)
    if case.pattern == "rear":
        ax.set_ylim(case.truth.center[0]*100-24, case.truth.center[0]*100+9)
    if zoom:
        ax.set_ylim(case.truth.center[0]*100-extent, case.truth.center[0]*100+extent)
        ax.figure.colorbar(mesh, ax=ax, shrink=.8, label="Density (1/cm²)")
    ax.set_xlabel("Lateral Y (cm)", fontsize=8)
    ax.set_ylabel("Depth X (cm)", fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/centroid_belief")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render-only", action="store_true", help="Replot saved cases without running inference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = standard_cases()
    if args.limit:
        cases = cases[:args.limit]
    records, displays = [], []
    run_seed, run_samples = args.seed, args.samples
    for index, case in enumerate(cases):
        if args.render_only:
            detail = json.loads((args.output / f"{case.name}.json").read_text())
            if index == 0:
                run_seed = detail["config"]["seed"]
                run_samples = detail["config"]["samples_per_shape"]
            row = detail["metrics"]
            with np.load(args.output / f"{case.name}.npz") as saved:
                posterior = Posterior(saved["centers"], saved["weights"], saved["shape_indices"],
                    tuple(s.name for s in SHAPES), detail["config"]["bandwidth"],
                    row["best_rms_cm"]/100, row["evaluations"],
                    tuple(filter(None, row.get("warnings", "").split(";"))))
                grid = {name: saved[name] for name in ("x", "y", "density", "threshold", "map")}
            records.append(row)
            displays.append((case, posterior, grid, row))
            continue
        config = Config(samples_per_shape=args.samples, seed=args.seed+index, complete_prior=0)
        start = time.perf_counter()
        posterior = estimate(case.observations, case.target_id, config=config)
        inference_seconds = time.perf_counter()-start
        grid = posterior.grid(step=.001)
        truth = np.asarray(case.truth.center[:2])
        observed = next(o.box for o in case.observations if o.observation_id == case.target_id)
        row = {"case": case.name, "shape": case.shape, "pattern": case.pattern,
               "observed_error_cm": float(np.linalg.norm(np.asarray(observed.center[:2])-truth)*100),
               "mean_error_cm": float(np.linalg.norm(posterior.mean-truth)*100),
               "map_error_cm": float(np.linalg.norm(grid["map"]-truth)*100),
               "covered90": bool(posterior.density(truth)[0] >= grid["threshold"]),
               "area90_cm2": grid["credible_area"]*10000,
               "ess": posterior.effective_samples, "best_rms_cm": posterior.best_rms*100,
               "inference_seconds": inference_seconds, "grid_integral": grid["integral"],
               "evaluations": posterior.evaluations, "warnings": ";".join(posterior.warnings)}
        records.append(row)
        displays.append((case, posterior, grid, row))
        np.savez_compressed(args.output / f"{case.name}.npz", centers=posterior.centers,
                            weights=posterior.weights, shape_indices=posterior.shape_indices,
                            x=grid["x"], y=grid["y"], density=grid["density"],
                            threshold=grid["threshold"], mean=posterior.mean, map=grid["map"],
                            covariance=posterior.covariance)
        detail = {"config": asdict(config), "input_observations": [asdict(o) for o in case.observations],
                  "target_id": case.target_id, "shape_posterior": posterior.shape_probabilities,
                  "evaluation_only_truth": asdict(case.truth), "metrics": row}
        (args.output / f"{case.name}.json").write_text(json.dumps(detail, indent=2)+"\n")
        print(f"[{index+1}/{len(cases)}] {case.name}: obs={row['observed_error_cm']:.2f} "
              f"mean={row['mean_error_cm']:.2f} cm, 90%={row['covered90']}, "
              f"ESS={row['ess']:.0f}, {inference_seconds:.1f}s", flush=True)
    with (args.output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, display in zip(axes.flat, displays[:4]):
        panel(ax, *display, zoom=True)
    for ax in list(axes.flat)[min(4, len(displays)):]:
        ax.axis("off")
    fig.suptitle("Centroid probability density: same 8-shape prior, no target shape label\n"
                 "*: true center; orange x: observed; red dot: mean; purple +: MAP; blue contour: 90% HPD\n"
                 "Center detail; foreground occluders are outside this crop (see full scene panels)", fontsize=11)
    fig.savefig(args.output / "centroid_detail.png", dpi=170)
    fig.savefig(args.output / "centroid_detail.pdf")
    plt.close(fig)
    for group in range((len(displays)+7)//8):
        selected = displays[group*8:(group+1)*8]
        panel_rows = 1 if len(selected) <= 4 else 2
        fig, axes = plt.subplots(panel_rows, 4, figsize=(15, 5 if panel_rows == 1 else 9), constrained_layout=True)
        for ax, display in zip(axes.flat, selected):
            panel(ax, *display)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.suptitle("OBB-only amodal centroid posterior | observer at +X\n"
                     "Dashed: full GT envelopes (evaluation only); orange: selected observed OBB; "
                     "*: truth; x: observed; red: mean; +: MAP; contour: 90% HPD", fontsize=11)
        fig.savefig(args.output / f"posterior_{group+1}.png", dpi=150)
        fig.savefig(args.output / f"posterior_{group+1}.pdf")
        plt.close(fig)
    standard = [r for r in records if "yaw0" in r["case"] and r["pattern"] in ("clear", "left", "right", "both")]
    summaries = {}
    for pattern in ("clear", "left", "right", "both"):
        rows = [r for r in standard if r["pattern"] == pattern]
        if rows:
            summaries[pattern] = {"count": len(rows), **{key: float(np.mean([r[key] for r in rows]))
                for key in ("observed_error_cm", "mean_error_cm", "map_error_cm", "covered90", "area90_cm2", "ess")}}
    (args.output / "summary.json").write_text(json.dumps({"seed": run_seed,
        "samples_per_shape": run_samples, "shapes": [asdict(s) for s in SHAPES],
        "by_pattern": summaries, "case_count": len(records)}, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    names = list(summaries)
    x = np.arange(len(names))
    axes[0].bar(x-.18, [summaries[n]["observed_error_cm"] for n in names], .36, label="Observed center")
    axes[0].bar(x+.18, [summaries[n]["mean_error_cm"] for n in names], .36, label="Posterior mean")
    axes[0].set_ylabel("Mean position error (cm)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [summaries[n]["covered90"]*100 for n in names])
    axes[1].axhline(90, color="red", ls="--", label="Nominal 90%")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("True center inside 90% HPD (%)")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [summaries[n]["area90_cm2"] for n in names])
    axes[2].set_ylabel("Mean 90% HPD area (cm²)")
    for ax in axes:
        ax.set_xticks(x, names)
    fig.suptitle("Eight predefined OBB proxies | fixed synthetic fixtures, not calibration evidence")
    fig.savefig(args.output / "summary.png", dpi=170)
    fig.savefig(args.output / "summary.pdf")
    plt.close(fig)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
