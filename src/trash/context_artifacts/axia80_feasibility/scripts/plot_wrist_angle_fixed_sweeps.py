"""Create fixed-Roll and fixed-Tilt response plots from an existing summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from axia80_feasibility.wrist_angle_plotting import (
    plot_wrist_angle_fixed_roll_responses,
    plot_wrist_angle_fixed_tilt_responses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    output_dir = args.output_dir or args.summary_csv.parent
    fixed_roll_path = output_dir / "axia80_wrist_angle_fixed_roll_vs_tilt.png"
    fixed_tilt_path = output_dir / "axia80_wrist_angle_fixed_tilt_vs_roll.png"
    plot_wrist_angle_fixed_roll_responses(args.summary_csv, fixed_roll_path, dpi=args.dpi)
    plot_wrist_angle_fixed_tilt_responses(args.summary_csv, fixed_tilt_path, dpi=args.dpi)
    print(fixed_roll_path.resolve())
    print(fixed_tilt_path.resolve())


if __name__ == "__main__":
    main()
