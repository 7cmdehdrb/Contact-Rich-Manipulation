"""Regenerate a cantilever force/bending plot from its summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axia80_feasibility.cantilever_plotting import plot_cantilever_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    output = args.output or args.csv.with_suffix(".png")
    plot_cantilever_summary(args.csv, output, dpi=args.dpi)
    print(f"[INFO]: Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
