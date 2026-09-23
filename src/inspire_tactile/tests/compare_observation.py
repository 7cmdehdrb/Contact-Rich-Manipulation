"""Compare the two logs produced by observation_probe.py; exit nonzero on dynamics changes."""

import argparse
import json
from pathlib import Path

import numpy as np


def samples(path):
    with path.open() as stream:
        return [row for line in stream if (row := json.loads(line)).get("type") == "sample"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("projected", type=Path)
    args = parser.parse_args()
    raw, projected = samples(args.raw), samples(args.projected)
    if not raw or len(raw) != len(projected):
        raise RuntimeError("Expected equal nonzero sample counts")
    if any(a["time_s"] != b["time_s"] for a, b in zip(raw, projected)):
        raise RuntimeError("Different sampling times")
    if any(a["force_source"] != "sensor_body_raw" or b["force_source"] != "contact_region_projection"
           for a, b in zip(raw, projected)):
        raise RuntimeError("Expected raw, then projected logs")
    fields = {
        "hand_rad": lambda r: [r["actual_rad"][n] for n in sorted(r["actual_rad"])],
        "arm_rad": lambda r: [r["arm_actual_rad"][n] for n in sorted(r["arm_actual_rad"])],
        "object_state": lambda r: r["object_state_w"],
        "raw_sensor_force_N": lambda r: r["raw_sensor_force_w_N"],
    }
    maxima = {}
    for name, getter in fields.items():
        a, b = (np.asarray([getter(row) for row in rows]) for rows in (raw, projected))
        maxima[name] = float(np.max(np.abs(a - b)))
        if not np.isfinite(a).all() or not np.array_equal(a, b):
            raise RuntimeError(f"{name}: trajectories differ; max abs difference={maxima[name]}")
    print(json.dumps({"steps": len(raw), "max_abs_difference": maxima}))
    # This is an exact regression on this machine, not a promise of bitwise
    # reproducibility across GPU/driver/PhysX versions.


if __name__ == "__main__":
    main()
