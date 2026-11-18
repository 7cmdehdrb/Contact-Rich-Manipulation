#!/usr/bin/env python3
"""CPU fixtures -> inference once -> saved five-number Gaussian -> plots."""
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
from centroid_belief import Config, GaussianBelief, COMPLETE_HYPOTHESIS, estimate_particles
from centroid_belief.fixtures import Case, standard_cases


def panel(ax, case, belief, row, *, context=False):
    observed = next(o.box for o in case.observations if o.observation_id == case.target_id)
    extent = max(.07, np.ptp(_footprint(case.truth)[:, 1])*.75)
    xx = np.linspace(case.truth.center[0]-extent, case.truth.center[0]+extent, 200)
    yy = np.linspace(case.truth.center[1]-extent, case.truth.center[1]+extent, 200)
    gx, gy = np.meshgrid(xx, yy)
    density = belief.density(np.column_stack((gx.ravel(), gy.ravel()))).reshape(gx.shape)
    mesh = ax.pcolormesh(yy*100, xx*100, density.T/10000, cmap="Blues", shading="auto")
    if context:
        boxes = case.scene
    else:
        boxes = (case.truth,)
    for box in boxes:
        ax.add_patch(Polygon(_footprint(box)[:, ::-1]*100, fill=False, ls="--", ec="#666666", lw=1.1))
    for o in case.observations:
        if context or o.observation_id == case.target_id:
            ax.add_patch(Polygon(_footprint(o.box)[:, ::-1]*100, fill=False,
                         ec="#e18b00" if o.observation_id == case.target_id else "#777777", lw=1.6))
    ellipse = belief.ellipse(.9)
    ax.plot(ellipse[:, 1]*100, ellipse[:, 0]*100, color="#1f527f", lw=1.6)
    ax.plot(case.truth.center[1]*100, case.truth.center[0]*100, "*", color="black", ms=10)
    ax.plot(observed.center[1]*100, observed.center[0]*100, "x", color="#e18b00", ms=8, mew=2)
    ax.plot(belief.mean[1]*100, belief.mean[0]*100, "o", color="#d33242", ms=5)
    ax.set_xlim(yy[0]*100, yy[-1]*100)
    ax.set_ylim(xx[0]*100, xx[-1]*100)
    if context:
        footprints = np.concatenate([_footprint(b) for b in case.scene])
        ax.set_xlim(min(yy[0], footprints[:, 1].min()-.01)*100,
                    max(yy[-1], footprints[:, 1].max()+.01)*100)
        ax.set_ylim(min(xx[0], footprints[:, 0].min()-.01)*100,
                    max(xx[-1], footprints[:, 0].max()+.01)*100)
    else:
        ax.figure.colorbar(mesh, ax=ax, shrink=.75, label="Density (1/cm²)")
    name = case.shape.split("_", 1)[-1] if case.shape[:3].isdigit() else case.shape
    ax.set_title(f"{name} | {case.pattern}\n"
                 f"Complete weight {row['complete_weight']:.1%}; observed X in 90%: {row['observed_ellipse90']}\n"
                 f"Center error {row['observed_error_cm']:.2f} -> {row['mean_error_cm']:.2f} cm",
                 fontsize=9)
    if row["warnings"]:
        ax.text(.02, .02, row["warnings"], transform=ax.transAxes, color="#a4262c", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral Y (cm)")
    ax.set_ylabel("Depth X (cm)")
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/centroid_gaussian")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--complete-prior", type=float, default=.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse saved cases with identical inputs and config")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = standard_cases()
    original = cases[2]  # Same OBB input, alternative complete-object ground truth.
    full_box = original.observations[0].box
    cases.append(Case("same_observation_complete_right", "complete_box", "touching_full",
                      full_box, (full_box, *original.scene[1:]), original.observations, original.target_id))
    if args.limit:
        cases = cases[:args.limit]
    displays, records, configs = [], [], []
    for index, case in enumerate(cases):
        destination = args.output / f"{case.name}.json"
        seed_index = 2 if case.pattern == "touching_full" else index
        config = Config(samples_per_shape=args.samples, seed=args.seed+seed_index,
                        complete_prior=args.complete_prior)
        if args.render_only or (args.resume and destination.exists()):
            detail = json.loads(destination.read_text())
            if args.resume and (detail["config"] != asdict(config) or
                    detail["input_observations"] != json.loads(json.dumps([asdict(o) for o in case.observations]))):
                raise ValueError(f"Saved input/config differs for {case.name}; use a fresh output directory")
            row = detail["metrics"]
        else:
            start = time.perf_counter()
            particles = estimate_particles(case.observations, case.target_id, config=config)
            belief = GaussianBelief.from_particles(particles)
            elapsed = time.perf_counter()-start
            observed = next(o.box for o in case.observations if o.observation_id == case.target_id)
            center = np.asarray(observed.center[:2])
            truth = np.asarray(case.truth.center[:2])
            # This baseline is read for evaluation ONLY, never supplied to inference.
            baseline = ROOT / "outputs/centroid_belief" / f"{case.name}.json"
            old_error = json.loads(baseline.read_text())["metrics"]["mean_error_cm"] if baseline.exists() else None
            row = {"case": case.name, "shape": case.shape, "pattern": case.pattern,
                   "observed_error_cm": float(np.linalg.norm(center-truth)*100),
                   "old_template_mean_error_cm": old_error,
                   "mean_error_cm": float(np.linalg.norm(belief.mean-truth)*100),
                   "complete_weight": particles.shape_probabilities[COMPLETE_HYPOTHESIS],
                   "observed_ellipse90": bool(belief.contains(center)[0]),
                   "truth_ellipse90": bool(belief.contains(truth)[0]),
                   "observed_1cm_square_probability": belief.probability_in_rectangle(center-.005, center+.005),
                   "area90_cm2": float(belief.credible_area(.9)*10000),
                   "ess": particles.effective_samples, "best_rms_cm": particles.best_rms*100,
                   "seconds": elapsed, "warnings": ";".join(particles.warnings)}
            detail = {"model": "single_gaussian_with_complete_hypothesis",
                      "parameter_order": ["mu_x", "mu_y", "cov_xx", "cov_xy", "cov_yy"],
                      "parameters": belief.parameters, "config": asdict(config),
                      "input_observations": [asdict(o) for o in case.observations],
                      "target_id": case.target_id, "evaluation_only_truth": asdict(case.truth),
                      "hypothesis_weights": particles.shape_probabilities, "metrics": row}
            destination.write_text(json.dumps(detail, indent=2)+"\n")
            del particles
            print(f"[{index+1}/{len(cases)}] {case.name}: complete={row['complete_weight']:.1%}, "
                  f"obs90={row['observed_ellipse90']}, error={row['mean_error_cm']:.2f}cm, "
                  f"ESS={row['ess']:.0f}, {elapsed:.1f}s", flush=True)
        # All plotting and subsequent probability queries use ONLY five values.
        belief = GaussianBelief.from_parameters(detail["parameters"])
        displays.append((case, belief, row))
        records.append(row)
        configs.append(detail["config"])
    with (args.output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, display in zip(axes.flat, displays[:4]):
        panel(ax, *display)
    for ax in list(axes.flat)[min(4, len(displays)):]:
        ax.axis("off")
    fig.suptitle("Single Gaussian centroid belief | complete and partial hypotheses evaluated first\n"
                 "Orange box/X: observed OBB/center; dashed: original OBB (evaluation only)\n"
                 "Black star: true center; red dot: mean = MAP; blue ellipse: 90% probability mass", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"gaussian_detail.{suffix}", dpi=170)
    plt.close(fig)
    for group in range((len(displays)+7)//8):
        selected = displays[group*8:(group+1)*8]
        nrows = 1 if len(selected) <= 4 else 2
        fig, axes = plt.subplots(nrows, 4, figsize=(15, 5 if nrows == 1 else 9), constrained_layout=True)
        for ax, display in zip(axes.flat, selected):
            panel(ax, *display, context=True)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
        fig.suptitle("Single Gaussian | observer at +X | colors scaled per panel\n"
                     "*: truth; orange X: observed; red dot: mean = MAP; blue ellipse: 90%", fontsize=11)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"gaussian_scenes_{group+1}.{suffix}", dpi=150)
        plt.close(fig)
    standard = [r for r in records if r["case"].endswith("yaw0") and r["pattern"] in ("clear", "left", "right", "both")]
    summary = {}
    fields = ("observed_error_cm", "mean_error_cm", "complete_weight", "observed_ellipse90",
              "truth_ellipse90", "area90_cm2", "observed_1cm_square_probability", "seconds")
    for pattern in ("clear", "left", "right", "both"):
        rows = [r for r in standard if r["pattern"] == pattern]
        if rows:
            summary[pattern] = {"count": len(rows), **{f: float(np.mean([r[f] for r in rows])) for f in fields}}
    (args.output / "summary.json").write_text(json.dumps({"case_count": len(records),
        "first_case_config": configs[0], "by_pattern": summary}, indent=2)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    names = list(summary)
    positions = np.arange(len(names))
    axes[0].bar(positions-.18, [summary[n]["observed_error_cm"] for n in names], .36, label="Observed center")
    axes[0].bar(positions+.18, [summary[n]["mean_error_cm"] for n in names], .36, label="Single Gaussian mean")
    axes[0].set_ylabel("Mean error (cm)")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions-.18, [100*summary[n]["observed_ellipse90"] for n in names], .36, label="Observed center")
    axes[1].bar(positions+.18, [100*summary[n]["truth_ellipse90"] for n in names], .36, label="True center")
    axes[1].set_ylabel("Inside 90% ellipse (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[2].bar(positions, [summary[n]["area90_cm2"] for n in names])
    axes[2].set_ylabel("Mean 90% ellipse area (cm²)")
    for ax in axes:
        ax.set_xticks(positions, names)
    fig.suptitle("Eight OBB proxies | fixed noiseless fixtures; probabilities are not calibrated")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"summary.{suffix}", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
