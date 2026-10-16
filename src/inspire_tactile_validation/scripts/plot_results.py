"""Create reproducible scientific figures from a single tactile validation run."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


STATUS = ["detected", "reference_mismatch", "miscontact", "sensor_miss", "insufficient_load", "no_contact"]
LABEL = ["Target detected*", "Force-reference mismatch", "Off-target contact", "Sensor miss", "Insufficient load", "No contact"]
COLORS = ["#3379b7", "#d99b2b", "#a45690", "#d66045", "#a0a6ae", "#e5e7eb"]
LOCATIONS = ["center", "corner_xminus_yminus", "corner_xminus_yplus", "corner_xplus_yminus", "corner_xplus_yplus"]
LOC_LABELS = ["Center", "X− Y−", "X− Y+", "X+ Y−", "X+ Y+"]


def resolve_run(path):
    if (path / "summary.json").exists():
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    dated = [p for p in candidates if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", p.name)]
    if not candidates:
        raise ValueError(f"No completed reports under {path}")
    return max(dated or candidates, key=lambda p: p.stat().st_mtime)


def short_name(name):
    return name.removeprefix("inspire_").replace("_force_sensor", "").replace("_", " ").title()


def decorate(fig, title, subtitle):
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=18, weight="bold")
    fig.text(0.08, 0.925, subtitle, fontsize=10, color="#525b66")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/inspire_tactile_validation"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-with", type=Path, help="An earlier run, displayed separately as a protocol comparison.")
    args = parser.parse_args()
    run = resolve_run(args.input.resolve())
    output = args.output_dir or run / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    trials = pd.read_csv(run / "trials.csv")
    samples = pd.read_csv(run / "samples.csv")
    names = summary["metadata"]["channels"]
    labels = [f"{i:02d}  {short_name(name)}" for i, name in enumerate(names)]
    if len(trials) != summary["trial_count"] or trials.trial_id.duplicated().any():
        raise ValueError("Trial counts/IDs disagree with the saved summary")
    if Counter(trials.status) != Counter(summary["status_counts"]):
        raise ValueError("Status counts disagree with the saved summary")
    if set(samples.trial_id.unique()) != set(trials.trial_id):
        raise ValueError("Raw sample trial IDs differ from trials.csv")
    if not samples.groupby("trial_id").size().sort_index().equals(trials.set_index("trial_id").sample_count.sort_index()):
        raise ValueError("Raw frame counts differ from trials.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#bcc2ca", "text.color": "#283442",
                         "axes.labelcolor": "#283442", "savefig.facecolor": "white"})
    figures = []

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [0.9, 1.5]})
    decorate(fig, "Tactile validation · trial outcomes", f"{run.name}  |  {len(trials)} trials · {len(samples):,} frames · 17 channels · pads: {summary['metadata'].get('pad_layout', 'original')}")
    counts = [int((trials.status == s).sum()) for s in STATUS]
    active = [i for i, c in enumerate(counts) if c]
    ax.barh([LABEL[i] for i in active], [counts[i] for i in active], color=[COLORS[i] for i in active])
    ax.invert_yaxis(); ax.set_xlabel("Trials (all attempted trials)")
    for y, i in enumerate(active):
        ax.text(counts[i] + len(trials) * .015, y, f"{counts[i]} ({counts[i]/len(trials):.1%})", va="center")
    ax.set_xlim(0, max(counts) * 1.38); ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    left = np.zeros(17)
    for index, status in enumerate(STATUS):
        values = trials[trials.status == status].groupby("channel").size().reindex(range(17), fill_value=0).to_numpy()
        bx.barh(np.arange(17), values, left=left, color=COLORS[index], label=LABEL[index]); left += values
    bx.set_yticks(range(17), labels); bx.invert_yaxis(); bx.set_xlabel("Trials per channel"); bx.set_xlim(0, max(left) * 1.08)
    bx.grid(axis="x", alpha=.15); bx.set_axisbelow(True)
    valid_frame_rows = trials[trials.valid == True]
    loaded_frame_count = int(valid_frame_rows.target_loaded_sample_count.sum())
    detected_frame_count = loaded_frame_count - int(valid_frame_rows.missed_loaded_sample_count.sum())
    frame_note = f"Loaded-frame detection: {detected_frame_count}/{loaded_frame_count} ({detected_frame_count/loaded_frame_count:.2%})." if loaded_frame_count else "No valid loaded frames."
    fig.text(.08, .035, f"*Detected is a trial-any bit, not a clean pass. Clean passes: {summary['clean_detection_trial_count']}/{len(trials)}.\n"
             f"Trials with any force-reference mismatch: {summary.get('reference_mismatch_trial_count', 0)}/{len(trials)} (overlaps detected).\n"
             f"Overall criteria: {summary['criteria_status']}. Invalid trials are excluded from conditional detection rates.\n{frame_note}", fontsize=10)
    fig.subplots_adjust(top=.85, bottom=.19, left=.20, right=.97, wspace=.85)
    figures.append(("01_overview", fig))

    repeats = sorted(trials["repeat"].unique())
    grid = np.full((17, 5 * len(repeats)), np.nan)
    mismatch = np.zeros_like(grid, dtype=bool)
    for row in trials.itertuples():
        col = repeats.index(row.repeat) * 5 + LOCATIONS.index(row.location)
        if not np.isnan(grid[row.channel, col]):
            raise ValueError("Duplicate channel/location/repeat trial")
        grid[row.channel, col] = STATUS.index(row.status)
        mismatch[row.channel, col] = row.reference_mismatch_count > 0
    fig, ax = plt.subplots(figsize=(max(12, len(repeats) * 4), 10))
    decorate(fig, "Sensor × target location × repetition", "D: detected · F: reference mismatch · O: off-target · !: additional reference diagnostic")
    ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(-.5, 6.5), 6))
    codes = ["D", "F", "O", "M", "L", "N"]
    for i, j in zip(*np.where(np.isfinite(grid))):
        value = int(grid[i, j]); suffix = "!" if mismatch[i, j] and value != 1 else ""
        ax.text(j, i, codes[value] + suffix, ha="center", va="center", fontsize=9, color="white" if value in (0, 2, 3) else "#283442")
    ax.set_yticks(range(17), labels)
    ax.set_xticks(range(grid.shape[1]), [f"R{r+1}\n{loc}" for r in repeats for loc in LOC_LABELS], fontsize=9)
    for j in range(5, grid.shape[1], 5): ax.axvline(j - .5, color="white", lw=3)
    ax.legend(handles=[Patch(color=COLORS[i], label=f"{codes[i]}: {LABEL[i]}") for i in active], loc="upper left", bbox_to_anchor=(0, -.09), ncol=3, frameon=False)
    fig.subplots_adjust(top=.86, left=.23, right=.98, bottom=.17)
    figures.append(("02_target_locations", fig))

    confusion = pd.read_csv(run / "confusion.csv")
    if list(confusion.expected_channel) != names:
        raise ValueError("Confusion matrix channel order differs from metadata")
    matrix = confusion[names].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 11))
    decorate(fig, "Observed channels during verified loaded contact", "Trial-any detection fraction; denominator = valid loaded trials per target. Grey / NA = no valid trials.")
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#e8ebef")
    plot = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap)
    ax.set_yticks(range(17), [f"{labels[i]}  (n={int(confusion.valid_loaded_trials.iloc[i])})" for i in range(17)])
    ax.set_xticks(range(17), [f"{i:02d}" for i in range(17)]); ax.set_xlabel("Observed channel index (same order as rows)"); ax.set_ylabel("Target channel")
    for i in range(17):
        for j in range(17):
            if not np.isfinite(matrix[i, j]):
                if j == i: ax.text(j, i, "NA", ha="center", va="center", fontsize=8)
            elif matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center", fontsize=8, color="white" if matrix[i,j] > .6 else "#283442")
    fig.colorbar(plot, ax=ax, fraction=.035, pad=.03, label="Fraction of valid loaded trials")
    fig.subplots_adjust(top=.86, bottom=.1, left=.3, right=.93)
    figures.append(("03_channel_confusion", fig))

    # One fixed, interpretable comparison: first repeat's palm center and first detected trial.
    chosen = [int(trials.sort_values(["repeat", "channel", "trial_id"]).iloc[0].trial_id)]
    detected = trials[trials.status == "detected"]
    if len(detected):
        others = detected[~detected.trial_id.isin(chosen)]
        if len(others): chosen.append(int(others.iloc[0].trial_id))
    fig, axes = plt.subplots(2, len(chosen), figsize=(7.3 * len(chosen), 9), squeeze=False, sharex="col")
    decorate(fig, "Force reference and tactile signal · example trials", "Examples selected deterministically: first trial and first other detected trial. Other repetitions remain in the raw CSV.")
    for col, trial_id in enumerate(chosen):
        info = trials[trials.trial_id == trial_id].iloc[0]
        raw = samples[samples.trial_id == trial_id]
        upper, lower = axes[:, col]
        for name, caption, color, style in [("commanded_force_n", "Commanded force (cap)", "#8e969e", "--"),
                                            ("force_balance_n", "Inertial force balance", "#d99b2b", "-"),
                                            ("probe_force_n", "Probe contact report", "#3379b7", "-")]:
            upper.plot(raw.time_s, raw[name], label=caption, color=color, linestyle=style, lw=1.6)
        threshold = float(info.threshold_n)
        upper.axhline(threshold, color="#283442", ls=":", lw=1, label=f"Threshold {threshold:g} N")
        upper.set_ylabel("Force magnitude (N)"); upper.set_ylim(bottom=0); upper.grid(alpha=.15)
        upper.set_title(f"Trial {trial_id} · {short_name(info['name'])} · {info.location}\n{info.status}", loc="left", fontsize=11)
        lower.step(raw.time_s, raw["bit_" + info["name"]], where="post", color="#3379b7", label="Final tactile bit", lw=1.8)
        lower.step(raw.time_s, (raw["direct_n_" + info["name"]] >= threshold).astype(int), where="post", color="#a45690", label="Direct pad bit", linestyle="--")
        lower.set_ylim(-.12, 1.25); lower.set_yticks([0, 1]); lower.set_xlabel("Trial time (s)"); lower.set_ylabel("Target contact bit"); lower.grid(alpha=.15)
        for phase, group in raw.groupby("phase", sort=False):
            start, stop = group.time_s.min(), group.time_s.max() + summary["metadata"]["physics_dt"]
            if phase in ("approach", "hold"):
                for axis in (upper, lower): axis.axvspan(start, stop, color="#dfe8f0", alpha=.35, zorder=-1)
            lower.text((start+stop)/2, 1.12, phase, ha="center", fontsize=8)
        upper.legend(loc="upper left", fontsize=8); lower.legend(loc="lower right", fontsize=8)
    baseline_reference = samples[samples.phase == "baseline"].force_balance_n.median()
    fig.text(.08, .035, f"Median no-contact force balance: {baseline_reference:.4g} N. Force balance is a diagnostic, not calibrated contact-force ground truth.", fontsize=9, color="#525b66")
    fig.subplots_adjust(top=.81, bottom=.14, left=.08, right=.98, hspace=.35, wspace=.25)
    figures.append(("04_example_time_series", fig))

    hold = samples[samples.phase == "hold"].groupby("trial_id")[["force_balance_n", "probe_force_n"]].median().reset_index().merge(trials[["trial_id", "status"]], on="trial_id")
    fig, ax = plt.subplots(figsize=(11, 8))
    decorate(fig, "Hold phase · reported contact vs force balance", "One point per trial: median hold force. Force balance is a diagnostic, not calibrated ground truth.")
    for i in active:
        group = hold[hold.status == STATUS[i]]
        ax.scatter(group.force_balance_n, group.probe_force_n, s=32, alpha=.65, color=COLORS[i], label=f"{LABEL[i]} (n={len(group)})", edgecolor="white", linewidth=.4)
    limit = max(hold.force_balance_n.max(), hold.probe_force_n.max(), .05) * 1.08
    ax.plot([0, limit], [0, limit], color="#283442", ls="--", lw=1, label="Equal magnitudes")
    ax.axhline(.05, color="#283442", ls=":", lw=1, label="0.05 N threshold")
    ax.set_yscale("symlog", linthresh=1e-5); ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel("Median inertial force balance (N)"); ax.set_ylabel("Median probe contact report (N) · symlog scale")
    ax.grid(alpha=.16); ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.12, left=.12, right=.96)
    figures.append(("05_hold_force_comparison", fig))
    hold.to_csv(output / "hold_force_medians.csv", index=False)


    if args.compare_with:
        prior_path = resolve_run(args.compare_with.resolve())
        prior = json.loads((prior_path / "summary.json").read_text())
        fig, ax = plt.subplots(figsize=(13, 6))
        decorate(fig, "Before / after · recorded trial outcomes", f"Protocol comparison: physics timing / gravity corrected; after-pad layout: {summary['metadata'].get('pad_layout','original')}. This is not a sensor-accuracy benchmark.")
        totals = [prior["trial_count"], len(trials)]
        left = np.zeros(2)
        for index, status in enumerate(STATUS):
            values = [prior["status_counts"].get(status, 0), counts[index]]
            ax.barh([0, 1], values, left=left, color=COLORS[index], label=LABEL[index])
            for row, value in enumerate(values):
                if value:
                    ax.text(left[row]+value/2, row, str(value), va="center", ha="center", color="white" if index in (0,2,3) else "#283442")
            left += values
        ax.set_yticks([0,1], ["Before · original pads / old stepping", f"After · {summary['metadata'].get('pad_layout','original')} pads / single tick"])
        ax.invert_yaxis(); ax.set_xlim(0, max(totals)*1.02); ax.set_xlabel("Attempted trials")
        ax.legend(handles=[Patch(color=COLORS[i], label=LABEL[i]) for i in range(6) if prior["status_counts"].get(STATUS[i],0) or counts[i]], loc="upper left", bbox_to_anchor=(0,-.18), ncol=3, frameon=False)
        fig.text(.08, .05, f"Clean passes: before {prior['clean_detection_trial_count']}/{totals[0]}, after {summary['clean_detection_trial_count']}/{totals[1]}. Original Sweep assets/code were not modified.", fontsize=10)
        fig.subplots_adjust(top=.77,bottom=.31,left=.32,right=.97)
        figures.append(("06_before_after", fig))

    hold_raw = samples[samples.phase == "hold"]
    hold_rows = []
    for info in trials.itertuples():
        raw = hold_raw[hold_raw.trial_id == info.trial_id]
        hold_rows.append({"channel":info.channel,"location":info.location,
                          "load_fraction":float((raw.probe_force_n >= info.threshold_n).mean()),
                          "bit_fraction":float(raw["bit_"+info.name].mean())})
    coverage = pd.DataFrame(hold_rows).groupby(["channel","location"])[["load_fraction","bit_fraction"]].mean()
    fig, axes = plt.subplots(1,2,figsize=(13,10),sharey=True)
    decorate(fig, "Hold phase · load availability and tactile duty", "Each cell averages the repeat-level fraction of hold frames. A contact bit can correctly turn off when load falls below 0.05 N.")
    for axis, key, title in zip(axes, ["load_fraction","bit_fraction"], ["Probe load ≥ threshold","Target bit = 1"]):
        matrix=np.array([[coverage.loc[(ch,loc),key] if (ch,loc) in coverage.index else np.nan for loc in LOCATIONS] for ch in range(17)])
        im=axis.imshow(np.ma.masked_invalid(matrix),aspect="auto",vmin=0,vmax=1,cmap="Blues")
        axis.set_title(title,fontsize=11)
        axis.set_xticks(range(5),LOC_LABELS,fontsize=9)
        axis.set_yticks(range(17),labels)
        for ch in range(17):
            for col in range(5):
                if np.isfinite(matrix[ch,col]):axis.text(col,ch,f"{matrix[ch,col]:.0%}",ha="center",va="center",fontsize=8,color="white" if matrix[ch,col]>.6 else "#283442")
    fig.subplots_adjust(top=.86,bottom=.10,left=.23,right=.92,wspace=.15)
    fig.colorbar(im,ax=list(axes),fraction=.025,pad=.025,label="Fraction of hold frames")
    figures.append(("07_hold_load_and_bit",fig))

    with PdfPages(output / "tactile_validation_figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(output / (name + ".png"), dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    manifest = {"run": str(run), "trial_count": len(trials), "sample_count": len(samples),
                "status_counts": dict(Counter(trials.status)), "figures": [name + ".png" for name, _ in figures],
                "sources": [str(run / f) for f in ("summary.json", "trials.csv", "samples.csv", "confusion.csv")],
                "excluded_other_runs": True}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
