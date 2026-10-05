"""Read-only TensorBoard analysis; no simulator or environment changes required."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hand_push_analysis_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "logs/rsl_rl/hand_manipulation_push/2026-10-03_22-48-10"
acc = EventAccumulator(str(RUN), size_guidance={"scalars": 0}).Reload()
series = {tag: acc.Scalars(tag) for tag in acc.Tags()["scalars"]}
iteration_series = {tag: rows for tag, rows in series.items() if not tag.endswith("/time")}
steps = sorted({row.step for rows in iteration_series.values() for row in rows})
lookup = {tag: {row.step: row.value for row in rows} for tag, rows in iteration_series.items()}
with (OUT / "scalars.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    tags = sorted(iteration_series)
    writer.writerow(["iteration", *tags])
    for step in steps:
        writer.writerow([step, *(lookup[tag].get(step, "") for tag in tags)])

windows = {"first_100": (0, 99), "iterations_900_999": (900, 999),
           "checkpoint_1950_last_100": (1851, 1950), "last_100": (1860, 1959)}
summary = {"run": str(RUN), "iterations": len(steps), "first_iteration": min(steps),
           "last_iteration": max(steps), "num_envs": 2048, "rollout_steps": 36,
           "policy_dt_s": 0.02, "reward_log_denominator_s": 10.0,
           "note": "Reward means are means of per-iteration reset-batch averages, not an episode-count-weighted estimate.",
           "windows": {}, "all_run_ranges": {}}
for name, (lo, hi) in windows.items():
    values = {tag: [row.value for row in rows if lo <= row.step <= hi]
              for tag, rows in iteration_series.items()}
    means = {tag: float(np.mean(vals)) for tag, vals in values.items() if vals}
    contributions = {tag.removeprefix("Episode_Reward/"): mean * 10.0
                     for tag, mean in means.items() if tag.startswith("Episode_Reward/")}
    positive = sum(max(value, 0) for value in contributions.values())
    negative = sum(abs(min(value, 0)) for value in contributions.values())
    summary["windows"][name] = {"start": lo, "end": hi, "means": means,
        "episode_reward_contributions": contributions,
        "positive_share": {key: max(value, 0) / positive if positive else 0 for key, value in contributions.items()},
        "negative_share": {key: abs(min(value, 0)) / negative if negative else 0 for key, value in contributions.items()},
        "sum_episode_contributions": sum(contributions.values())}
for tag, rows in iteration_series.items():
    vals = np.array([row.value for row in rows])
    summary["all_run_ranges"][tag] = {"min": float(vals.min()), "max": float(vals.max()),
        "nonzero_iterations": int(np.count_nonzero(vals)), "last": float(vals[-1])}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
with (OUT / "reward_windows.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["window", "term", "episode_contribution", "positive_share", "negative_share"])
    for name, data in summary["windows"].items():
        for term, value in data["episode_reward_contributions"].items():
            writer.writerow([name, term, value, data["positive_share"][term], data["negative_share"][term]])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
def smoothed(tag, width=50, scale=1):
    rows = iteration_series[tag]
    x = np.array([row.step for row in rows])
    y = np.array([row.value * scale for row in rows])
    return x[width-1:], np.convolve(y, np.ones(width) / width, mode="valid")
for key in ["approach", "progress", "first_contact", "contact", "alignment", "action_rate", "backslide"]:
    x, y = smoothed("Episode_Reward/" + key, scale=10)
    axes[0].plot(x, y, label=key)
axes[0].axhline(0, color="grey", lw=0.6)
axes[0].set(title="Weighted shaping reward contribution per episode (50-iteration average)", ylabel="Episode reward")
axes[0].legend(ncol=4)
for key in ["contact_seen", "push_seen", "success"]:
    x, y = smoothed("Metrics/target_position/" + key)
    axes[1].plot(x, y, label=key)
axes[1].set(title="Contact occurs, but pushing is rare and no successes are logged", ylabel="Logged fraction", ylim=(-0.02, 1.02))
ax_distance = axes[1].twinx()
x, y = smoothed("Metrics/target_position/goal_distance_m")
ax_distance.plot(x, y, color="grey", ls="--", label="goal distance")
ax_distance.set_ylabel("Goal distance (m)")
ax_distance.set_ylim(0, 0.32)
lines = axes[1].get_lines() + ax_distance.get_lines()
axes[1].legend(lines, [line.get_label() for line in lines], loc="center right")
x, y = smoothed("Policy/mean_std", width=1)
axes[2].plot(x, y, color="#bd3c26", label="Gaussian mean std")
axes[2].axhline(1, color="grey", ls="--", label="Executed normalized action bound: +/-1")
axes[2].set(title="Raw Gaussian exploration scale grows far beyond the executed action range", ylabel="Action standard deviation", xlabel="PPO iteration", yscale="log")
axes[2].legend()
fig.savefig(OUT / "training_diagnosis.png", dpi=170)
plt.close(fig)
print(json.dumps({name: data for name, data in summary["windows"].items()}, indent=2))
print("Selected full-run ranges:")
for tag in ["Metrics/target_position/push_seen", "Metrics/target_position/success", "Episode_Reward/progress", "Episode_Reward/failure", "Episode_Termination/failure", "Train/mean_reward", "Policy/mean_std"]:
    print(tag, summary["all_run_ranges"][tag])
print("Artifacts:", OUT)
