"""Regenerate wrist Roll x Tilt figures from a summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from axia80_feasibility.wrist_angle_plotting import (
    plot_wrist_angle_center_slices,
    plot_wrist_angle_error_heatmaps,
    plot_wrist_angle_heatmaps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help="Prefix for *_response_heatmaps.png, *_error_heatmaps.png, and *_center_slices.png.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    prefix = args.output_prefix or args.summary_csv.with_name("axia80_wrist_angle")
    response = prefix.with_name(prefix.name + "_response_heatmaps.png")
    error = prefix.with_name(prefix.name + "_error_heatmaps.png")
    slices = prefix.with_name(prefix.name + "_center_slices.png")
    plot_wrist_angle_heatmaps(args.summary_csv, response, dpi=args.dpi)
    plot_wrist_angle_error_heatmaps(args.summary_csv, error, dpi=args.dpi)
    plot_wrist_angle_center_slices(args.summary_csv, slices, dpi=args.dpi)
    print(response)
    print(error)
    print(slices)


if __name__ == "__main__":
    main()
