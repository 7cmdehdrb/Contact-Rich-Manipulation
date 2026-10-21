#!/usr/bin/env python3
"""Export per-axis backward finite differences of compensated F/T.

World-axis rates avoid changes caused solely by EEF-axis rotation. EEF-axis
rates are also exported separately, as differences of rotating-frame components.
No derivative crosses environments or episode boundaries. No smoothing/clipping.
"""

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

AXES = ("fx", "fy", "fz", "mx", "my", "mz")
CONTEXT = ("env_id", "episode_id", "control_step", "episode_step", "sim_time_s",
           "episode_time_s", "terminated", "time_out", "tactile_count", "goal_error_xy_m")


def integer(row, name):
    value = float(row[name])
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"Invalid integer field {name}: {value}")
    return int(value)


def process(source_dir, output_dir, frames=("world", "eef")):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source = source_dir / "wrench.csv"
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    if source_metadata["status"] != "complete":
        raise ValueError("Source acquisition did not complete")
    dt = float(source_metadata["control_dt_s"])
    physics_dt = float(source_metadata["physics_dt_s"])
    if not all(math.isfinite(x) and x > 0 for x in (dt, physics_dt)):
        raise ValueError("Invalid source timestep")
    with source.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    manifest = source_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["csv_sha256"] != source_hash:
        raise ValueError("Source CSV hash differs from acquisition manifest")
    if not frames or len(set(frames)) != len(frames) or set(frames) - {"world", "eef"}:
        raise ValueError("Frames must be a nonempty subset of world/eef")
    if output_dir == source_dir or output_dir in source_dir.parents:
        raise ValueError("Use a separate derived-output directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_csv=str(source), source_sha256=source_hash,
        control_dt_s=dt, frames=list(frames), input_rows=0, output_rows_per_file=0,
        excluded_invalid_sample_rows=0, excluded_invalid_delta_rows=0,
        method="backward finite difference: (current value - previous same-env/episode value) / control_dt_s",
        units={"force_rate":"N/s", "moment_rate":"N*m/s"},
        moment_origin="EEF; world denotes axes, not moment origin",
        eef_interpretation="difference of rotating EEF-frame components, includes frame rotation",
        world_interpretation="difference of world-frame components, EEF moment reference still moves",
        timestamp="rate corresponds to interval ending at the current row",
        sample_time="counter-based control interval, not wall clock or rounded float32 sim_time_s",
        processing="signed and absolute rates; no smoothing, clipping, normalization or outlier removal",
        files=[],
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    previous = {}
    try:
        with ExitStack() as stack:
            writers = {}
            for frame in frames:
                (output_dir / frame).mkdir()
                for axis in AXES:
                    unit = "N" if axis.startswith("f") else "Nm"
                    fields = ["source_data_row", *CONTEXT, "previous_control_step", "delta_time_s",
                              f"previous_value_{unit}", f"value_{unit}", f"delta_{unit}",
                              f"rate_{unit}_per_s", f"abs_rate_{unit}_per_s"]
                    relative = f"{frame}/{axis}_rate.csv"
                    file = stack.enter_context((output_dir / relative).open("x", newline=""))
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    writers[frame, axis] = writer
                    report["files"].append(relative)
            file = stack.enter_context(source.open(newline=""))
            reader = csv.DictReader(file)
            for number, raw in enumerate(reader, start=1):
                row = {key:float(value) for key,value in raw.items()}
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError(f"Nonfinite input at data row {number}")
                env = integer(row, "env_id")
                step = integer(row, "control_step")
                episode = integer(row, "episode_id")
                last = previous.get(env)
                if last is not None and step != last["control_step"] + 1:
                    raise ValueError(f"Missing/duplicate/unordered step for env {env}, step {step}")
                if row["sample_valid"] not in (0,1) or row["delta_valid"] not in (0,1):
                    raise ValueError("Validity flags must be 0 or 1")
                report["input_rows"] += 1
                if not row["sample_valid"]:
                    report["excluded_invalid_sample_rows"] += 1
                elif not row["delta_valid"]:
                    report["excluded_invalid_delta_rows"] += 1
                else:
                    if (last is None or episode != last["episode_id"] or not last["sample_valid"]
                            or last["terminated"] or last["time_out"]):
                        raise ValueError(f"delta_valid crosses an invalid/reset boundary at data row {number}")
                    if (row["episode_step"] != last["episode_step"] + 1
                            or not math.isclose((row["physics_step"] - last["physics_step"])*physics_dt,
                                                dt, abs_tol=1e-10, rel_tol=1e-10)):
                        raise ValueError(f"Inconsistent sample interval at data row {number}")
                    context = {key:row[key] for key in CONTEXT}
                    for key in ("env_id", "episode_id", "control_step", "episode_step", "terminated", "time_out", "tactile_count"):
                        context[key] = integer(row, key)
                    for frame in frames:
                        for axis in AXES:
                            unit = "N" if axis.startswith("f") else "Nm"
                            key = f"v2_pure_{frame}_{axis}_{unit}"
                            delta = row[key] - last[key]
                            rate = delta / dt
                            if not math.isfinite(rate):
                                raise ValueError("Nonfinite derivative")
                            writers[frame,axis].writerow(context | {
                                "source_data_row":number, "previous_control_step":int(last["control_step"]),
                                "delta_time_s":dt, f"previous_value_{unit}":last[key], f"value_{unit}":row[key],
                                f"delta_{unit}":delta, f"rate_{unit}_per_s":rate, f"abs_rate_{unit}_per_s":abs(rate),
                            })
                    report["output_rows_per_file"] += 1
                previous[env] = row
        if report["input_rows"] != source_metadata["csv_rows"]:
            raise ValueError("Input row count differs from metadata")
        if not report["output_rows_per_file"]:
            raise ValueError("No valid derivative samples")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", choices=("world", "eef"), nargs="+", default=["world", "eef"])
    args = parser.parse_args()
    print(json.dumps(process(args.source_dir, args.output_dir, tuple(args.frames)), indent=2))
