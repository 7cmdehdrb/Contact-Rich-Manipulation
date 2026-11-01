"""Centre true object bounds on a regular grid inside the Final shelf."""

from dataclasses import dataclass
import math


KINDS = ("cylinder", "cube", "mug", "bottle",
         "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "002_master_chef_can")


@dataclass(frozen=True)
class Placement:
    kind: str
    row: int
    col: int
    position: tuple[float, float, float]


def grid_layout(rows, cols, bounds, *, gap=0.05, margin=0.02,
                shelf_half_extents=(0.20, 0.50), surface_height=1.05):
    """Rows run front-to-back (+X to -X), columns left-to-right (-Y to +Y).

    ``bounds`` maps each kind to its unscaled (lower, upper) local bounds.
    Cell pitches use the largest occupied footprint; no geometry is shrunk.
    Returned positions are asset roots, corrected for asymmetric bounds.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (rows, cols)):
        raise ValueError("rows and cols must be positive integers")
    if not all(math.isfinite(v) for v in (gap, margin, surface_height, *shelf_half_extents)):
        raise ValueError("Grid parameters must be finite")
    if gap < 0 or margin < 0 or any(h <= margin for h in shelf_half_extents):
        raise ValueError("Use nonnegative gap/margin and a shelf larger than its margins")
    kinds = [KINDS[i % len(KINDS)] for i in range(rows * cols)]
    for kind in set(kinds):
        low, high = bounds[kind]
        if len(low) != 3 or len(high) != 3 or not all(
            math.isfinite(a) and math.isfinite(b) and b > a for a, b in zip(low, high)
        ):
            raise ValueError(f"Invalid bounds for {kind}")
    size_x = max(bounds[k][1][0] - bounds[k][0][0] for k in kinds)
    size_y = max(bounds[k][1][1] - bounds[k][0][1] for k in kinds)
    extent = (rows * size_x + (rows - 1) * gap, cols * size_y + (cols - 1) * gap)
    available = tuple(2 * (h - margin) for h in shelf_half_extents)
    if any(size > room + 1e-9 for size, room in zip(extent, available)):
        raise ValueError(
            f"{rows}x{cols} grid needs {extent[0]:.3f} x {extent[1]:.3f} m; "
            f"shelf allows {available[0]:.3f} x {available[1]:.3f} m. "
            "Reduce rows/cols or --gap; object scale remains 1.0."
        )
    result = []
    for i, kind in enumerate(kinds):
        row, col = divmod(i, cols)
        low, high = bounds[kind]
        center_x = ((rows - 1) / 2 - row) * (size_x + gap)
        center_y = (col - (cols - 1) / 2) * (size_y + gap)
        result.append(Placement(kind, row, col, (
            center_x - (low[0] + high[0]) / 2,
            center_y - (low[1] + high[1]) / 2,
            surface_height - low[2],
        )))
    return result
