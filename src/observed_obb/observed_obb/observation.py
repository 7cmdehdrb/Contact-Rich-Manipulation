"""Geometry-only snapshot boundary. Coordinates and dimensions are in metres."""
from dataclasses import dataclass
from collections.abc import Sequence
import math

import numpy as np


@dataclass(frozen=True)
class Box:
    """Upright box, centred on geometry (not necessarily the asset root)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("Box center and size must each have three components")
        if not all(math.isfinite(v) for v in (*self.center, *self.size, self.yaw)):
            raise ValueError("Box geometry must be finite")
        if min(self.size) <= 0:
            raise ValueError("Box dimensions must be positive")
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "size", tuple(float(v) for v in self.size))


@dataclass(frozen=True)
class Observation:
    """Snapshot-local identity; no physical-source identity or semantic role."""

    observation_id: int
    box: Box


def _footprint(box: Box) -> np.ndarray:
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    rotation = np.array(((c, -s), (s, c)))
    corners = np.array(((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)))
    return (corners * box.size[:2]) @ rotation.T + box.center[:2]


def _front_depth(polygon: np.ndarray, lateral: float) -> float:
    depths = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if min(a[0], b[0]) <= lateral <= max(a[0], b[0]) and a[0] != b[0]:
            depths.append(a[1] + (lateral - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
    return min(depths, default=math.inf)


def _visibility_edges(polygons: list[np.ndarray]) -> np.ndarray:
    """Partition where a boundary starts, bends, ends, or changes depth order."""
    cuts = [float(u) for polygon in polygons for u in polygon[:, 0]]
    segments = [(a, b) for polygon in polygons
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]
    for index, (a, b) in enumerate(segments):
        delta = b-a
        for c, d in segments[index+1:]:
            other = d-c
            determinant = delta[0]*other[1] - delta[1]*other[0]
            if determinant == 0:
                continue
            offset = c-a
            t = (offset[0]*other[1] - offset[1]*other[0]) / determinant
            u = (offset[0]*delta[1] - offset[1]*delta[0]) / determinant
            if 0 < t < 1 and 0 < u < 1:
                cuts.append(float(a[0] + t*delta[0]))
    return np.unique(cuts)


def _clip(polygon: np.ndarray, bound: float, keep_greater: bool) -> np.ndarray:
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        a_in = a[0] >= bound if keep_greater else a[0] <= bound
        b_in = b[0] >= bound if keep_greater else b[0] <= bound
        if a_in:
            result.append(a)
        if a_in != b_in:
            t = (bound - a[0]) / (b[0] - a[0])
            result.append(a + t * (b - a))
    return np.asarray(result)


def _fit(polygon: np.ndarray, source: Box) -> Box:
    # A minimum-area enclosing rectangle of a convex polygon has an edge
    # parallel to a polygon edge. Clipping a box leaves at most six vertices.
    # Work near the origin to limit cancellation in translated scenes.
    origin = polygon.mean(axis=0)
    local = polygon - origin
    candidates = []
    for edge in np.roll(local, -1, axis=0) - local:
        if np.linalg.norm(edge) == 0:
            continue
        yaw = math.atan2(edge[1], edge[0]) % (math.pi / 2)
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.array(((c, s), (-s, c)))
        projected = local @ axes.T
        low, high = projected.min(axis=0), projected.max(axis=0)
        size = high - low
        candidates.append((float(size.prod()), yaw, axes, low, high))
    # Stable edge order resolves exact ties; no random direction selection.
    _, yaw, axes, low, high = min(candidates, key=lambda item: item[0])
    center = (low + high) / 2 @ axes + origin
    return Box((*center, source.center[2]), (*(high-low), source.size[2]), yaw)


def observe(
    boxes: Sequence[Box], *, direction: tuple[float, float] = (-1.0, 0.0),
    minimum_width: float = 0.02, minimum_ratio: float = 0.20,
) -> tuple[Observation, ...]:
    """Generate a fixed frontal snapshot. Direction points from viewer into scene.

    All inputs are occluders regardless of whether they survive the filter.
    Source identities and roles never enter the output. The default observes
    from shelf +X toward -X, with lateral width along shelf Y.
    """
    forward = np.asarray(direction, dtype=float)
    if forward.shape != (2,) or not np.isfinite(forward).all() or np.linalg.norm(forward) == 0:
        raise ValueError("Observation direction must be a finite nonzero XY vector")
    if not math.isfinite(minimum_width) or minimum_width <= 0:
        raise ValueError("Minimum width must be finite and positive")
    if not math.isfinite(minimum_ratio) or not 0 <= minimum_ratio <= 1:
        raise ValueError("Minimum ratio must be in [0, 1]")
    forward /= np.linalg.norm(forward)
    frame = np.array(((forward[1], -forward[0]), forward))
    polygons = [_footprint(box) @ frame.T for box in boxes]
    if not polygons:
        return ()
    edges = _visibility_edges(polygons)
    intervals: list[list[tuple[float, float]]] = [[] for _ in boxes]
    for lo, hi in zip(edges[:-1], edges[1:]):
        depths = [_front_depth(p, (lo + hi) / 2) for p in polygons]
        owner = int(np.argmin(depths))
        if math.isinf(depths[owner]):
            continue
        parts = intervals[owner]
        if parts and parts[-1][1] == lo:
            parts[-1] = (parts[-1][0], hi)
        else:
            parts.append((lo, hi))
    observations: list[Observation] = []
    for box, polygon, parts in zip(boxes, polygons, intervals):
        low, high = polygon[:, 0].min(), polygon[:, 0].max()
        threshold = max(minimum_width, minimum_ratio * (high - low))
        for lo, hi in parts:
            # Only account for subtraction roundoff, not a physical tolerance.
            rounding = 4 * np.finfo(float).eps * max(abs(lo), abs(hi), threshold)
            if hi - lo < threshold - rounding:
                continue
            if lo == low and hi == high:
                fitted = box
            else:
                clipped = _clip(_clip(polygon, lo, True), hi, False) @ frame
                fitted = _fit(clipped, box)
            observations.append(Observation(len(observations), fitted))
    return tuple(observations)
