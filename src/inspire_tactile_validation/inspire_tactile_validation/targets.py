"""CPU-only probe targets in the production tactile pad coordinate frames.

The source geometry is read from the active Sweep package without importing
Isaac Lab or editing USD files. Each pad has one center target and four corner
targets. A corner follows the center-to-bounding-box-corner ray to the actual
convex footprint boundary, then moves inward by ``corner_inset`` of that ray.
Thus a clipped polygon corner never produces an out-of-footprint test point.

Target Z is the outermost pad collision vertex, not the parent contact region's
maximum Z. Several pads are buried in the existing parent collision hull: the
physical probe can contact that hull before reaching the nominal pad surface.
This is intentional, because the production reader maps those parent contacts
back into the same tactile regions.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path


DEFAULT_REGION_PATH = (
    Path(__file__).resolve().parents[2]
    / "inspire_sweep_baseline/inspire_sweep_baseline/assets/data/tactile_regions.json"
)


def load_regions(path: str | Path | None = None) -> tuple[dict, ...]:
    """Load the unchanged production metadata in its policy channel order."""
    metadata = json.loads((DEFAULT_REGION_PATH if path is None else Path(path)).read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Expected tactile region metadata schema_version=1")
    if metadata.get("coordinate_frame") != "sensor_body_local":
        raise ValueError("Tactile regions must use sensor_body_local coordinates")
    regions = tuple(metadata["regions"])
    names = [region["name"] for region in regions]
    if len(regions) != 17 or len(set(names)) != len(names):
        raise ValueError("Expected exactly 17 uniquely named tactile regions")
    return regions


def _ray_corner(center: Sequence[float], corner: Sequence[float], planes: Sequence[Sequence[float]], inset: float):
    direction = [corner[axis] - center[axis] for axis in range(2)]
    radius = 1.0
    for a, b, c in planes:
        distance = a * center[0] + b * center[1] + c
        if distance > 1.0e-9:
            raise ValueError("The tactile bounding-box center is outside its convex footprint")
        slope = a * direction[0] + b * direction[1]
        if slope > 1.0e-15:
            radius = min(radius, max(0.0, -distance / slope))
    if radius <= 0.0:
        raise ValueError("A tactile footprint must have nonzero area around its center")
    return [center[axis] + (1.0 - inset) * radius * direction[axis] for axis in range(2)]


def build_targets(corner_inset: float = 0.2, *, regions: Sequence[dict] | None = None) -> list[dict]:
    """Return 85 serializable targets: center plus four inset corners per pad.

    ``corner_inset`` is a fraction of the center-to-boundary distance, with
    0.2 placing a corner sample 20% inward from the footprint boundary. It must
    be strictly between zero and one, so all four corner samples remain inside
    the polygon and distinct from the center. Normals point along each sensor
    body's local +Z; callers transform points and normals by the live body pose.
    """
    corner_inset = float(corner_inset)
    if not math.isfinite(corner_inset) or not 0.0 < corner_inset < 1.0:
        raise ValueError("corner_inset must be finite and strictly between 0 and 1")
    regions = load_regions() if regions is None else regions
    result = []
    for channel, region in enumerate(regions):
        lower = region["pad_bounds_m"]["min"]
        upper = region["pad_bounds_m"]["max"]
        if len(lower) != 3 or len(upper) != 3 or not all(
            math.isfinite(value) for value in (*lower, *upper)
        ) or any(lower[axis] >= upper[axis] for axis in range(3)):
            raise ValueError(f"Invalid pad bounds for {region['name']}")
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(2)]
        normal = [float(value) for value in region["normal_axis"]]
        if normal != [0.0, 0.0, 1.0]:
            raise ValueError("Probe planning requires sensor-local +Z outward normals")
        locations = [("center", center)]
        for x_positive, y_positive in ((False, False), (False, True), (True, False), (True, True)):
            corner = [upper[0] if x_positive else lower[0], upper[1] if y_positive else lower[1]]
            label = f"corner_x{'plus' if x_positive else 'minus'}_y{'plus' if y_positive else 'minus'}"
            locations.append((label, _ray_corner(center, corner, region["footprint_halfspaces"], corner_inset)))
        for label, xy in locations:
            result.append({
                "channel": channel,
                "name": region["name"],
                "parent": region["parent_name"],
                "location": label,
                "local_point": [float(xy[0]), float(xy[1]), float(upper[2])],
                "normal": normal.copy(),
            })
    return result
