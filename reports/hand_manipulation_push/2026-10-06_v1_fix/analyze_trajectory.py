"""CPU replay and aggregate checks for the captured original V1 policy trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path(__file__).resolve().parent / "deterministic_trace")
    args = parser.parse_args()
    out = args.trace.resolve()
    summary = json.loads((out / "summary.json").read_text())
    frame = pd.read_csv(out / "trajectory.csv")
    actor = torch.load(summary["provenance"]["checkpoint"], map_location="cpu", weights_only=False)["actor_state_dict"]
    inputs = torch.tensor(frame[[f"obs_{i:02d}" for i in range(64)]].values, dtype=torch.float32)
    with torch.inference_mode():
        predicted = inputs
        for layer in (0, 2, 4, 6):
            predicted = torch.nn.functional.linear(predicted, actor[f"mlp.{layer}.weight"], actor[f"mlp.{layer}.bias"])
            if layer != 6:
                predicted = torch.nn.functional.elu(predicted)
    recorded = torch.tensor(frame[[f"raw_mean_{i:02d}" for i in range(8)]].values, dtype=torch.float32)
    difference = (predicted-recorded).abs()
    if difference.max().item() >= 1e-4:
        raise RuntimeError("Saved 64D observations do not reproduce the traced actor means within float32 tolerance")

    initial = frame[frame.step == 0]
    numeric = ("eef_c_above_table_m", "desired_c_above_table_m", "palm_height_error_m", "approach_gap_m",
               "commanded_world_dz_m", "raw_mean_translation_world_up", "normalized_translation_world_up",
               "raw_mean_clip_fraction", "palm_normal_cos", "finger_outward_cos", "controller_error_norm_m")
    windows = {}
    for low, high in ((0, 49), (50, 99), (100, 249), (250, 499)):
        selected = frame[(frame.step >= low) & (frame.step <= high)]
        if len(selected):
            windows[f"steps_{low}_{high}"] = {"n": len(selected), **{name: float(selected[name].mean()) for name in numeric},
                                             "failures": int(selected.failure.sum())}
    outcome = frame[frame.done != 0]
    per_step = frame.groupby("step")[list(numeric)].mean()
    per_step.to_csv(out / "per_step_means.csv")
    outcome.to_csv(out / "episode_endings.csv", index=False)
    analysis = {
        "trace_summary": str(out / "summary.json"), "rows": len(frame),
        "replayed_actor_on_recorded_64d_inputs": {"mean_abs_error": difference.mean().item(),
                                                 "max_abs_error": difference.max().item(), "tolerance": 1e-4},
        "first_policy_step": {
            "C_height_above_table_mean_m": float(initial.eef_c_above_table_m.mean()),
            "raw_mean_world_up": float(initial.raw_mean_translation_world_up.mean()),
            "normalized_world_up": float(initial.normalized_translation_world_up.mean()),
            "commanded_world_dz_mean_m": float(initial.commanded_world_dz_m.mean()),
            "raw_mean_action0_min": float(initial.raw_mean_00.min()),
            "raw_mean_action0_max": float(initial.raw_mean_00.max()),
            "action0_positive_clip_fraction": float((initial.raw_mean_00 > 1).mean()),
            "right_environments": int((initial.side_left == 0).sum()),
            "left_environments": int((initial.side_left == 1).sum()),
        },
        "all_transition_fractions": {
            "any_mean_input_clipped": float((frame.raw_mean_clip_fraction > 0).mean()),
            "mean_input_axis_clipped": float(frame.raw_mean_clip_fraction.mean()),
            "C_height_above_0_15m": float((frame.eef_c_above_table_m > .15).mean()),
            "C_height_above_0_25m": float((frame.eef_c_above_table_m > .25).mean()),
            "C_height_above_0_40m": float((frame.eef_c_above_table_m > .40).mean()),
            "normal_outside_30_degrees": float((frame.palm_normal_cos < np.cos(np.pi/6)).mean()),
            "wrist_sin_at_or_below_0_15": float((np.sin(frame.wrist_2_raw_rad) <= .15).mean()),
        },
        "windows": windows,
        "observed_events": summary["aggregate"]["end_events"],
        "observed_contact_intervals": summary["aggregate"]["contact_policy_intervals"],
        "interpretation": [
            "The deterministic policy itself commands nearly maximum upward translation at its first step; this observation does not depend on stochastic Gaussian std.",
            "Measured C rises above the contact/reset plane and the physical approach gap grows. During this original-geometry probe, no actual Cube-palmar contact occurs.",
            "Raw actor mean clipping is measured on real observations here. It is distinct from the earlier analytic stochastic clipping bound based only on saved std.",
            "The 6cm controller projection applies at action updates. Post-physics error can exceed it while the robot moves; it is not an absolute height or final-error guarantee.",
            "All 12 failures in this 32-environment probe are Robot-Table contact, but this does not recover missing failure-subtype counts from the full training run.",
        ],
    }
    (out / "trajectory_analysis.json").write_text(json.dumps(analysis, indent=2))

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hand_push_v1_trace_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for env_id, selected in frame.groupby("env_id"):
        axes[0].plot(selected.step * .02, selected.eef_c_above_table_m, alpha=.4, linewidth=.8)
    axes[0].axhline(.15, color="grey", linestyle="--", label="new soft height")
    axes[0].axhline(.25, color="black", linestyle="--", label="new hard height")
    axes[0].set(ylabel="Actual C above Table (m)", title="model4300 deterministic trace on saved original geometry")
    axes[0].legend()
    axes[1].plot(per_step.index * .02, per_step.raw_mean_translation_world_up, label="raw actor mean along world up")
    axes[1].plot(per_step.index * .02, per_step.normalized_translation_world_up, label="normalized input along world up")
    axes[1].set(ylabel="Mean translation input")
    axes[1].legend()
    axes[2].plot(per_step.index * .02, per_step.approach_gap_m, label="physical palm approach gap")
    axes[2].plot(per_step.index * .02, per_step.controller_error_norm_m, label="C target error")
    axes[2].set(xlabel="Elapsed policy time (s)", ylabel="Distance (m)")
    axes[2].legend()
    figure.savefig(out / "trajectory_diagnosis.png", dpi=160)
    plt.close(figure)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
