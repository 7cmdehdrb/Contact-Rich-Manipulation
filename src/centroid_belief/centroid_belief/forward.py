"""Candidate-only adapter to the existing geometry observation kernel.

Other observed OBBs are approximate occlusion proxies. No physical collision
rejection is made from their overlap with a candidate.
"""
import math
import numpy as np
from observed_obb import Box
from observed_obb.observation import _footprint, _visibility_edges, _front_depth, _clip, _fit

FRAME = np.array(((0., 1.), (-1., 0.)))  # lateral Y, front-to-back -X


class ForwardModel:
    def __init__(self, occluders, minimum_width=.02, minimum_ratio=.20):
        self.polygons = [_footprint(box) @ FRAME.T for box in occluders]
        self.minimum_width = minimum_width
        self.minimum_ratio = minimum_ratio

    def predict(self, candidate: Box) -> tuple[Box, ...]:
        polygon = _footprint(candidate) @ FRAME.T
        polygons = [polygon, *self.polygons]
        edges = _visibility_edges(polygons)
        parts: list[tuple[float, float]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            u = (lo + hi) / 2
            depth = _front_depth(polygon, u)
            if not math.isfinite(depth):
                continue
            if any(_front_depth(other, u) < depth for other in self.polygons):
                continue
            if parts and parts[-1][1] == lo:
                parts[-1] = (parts[-1][0], hi)
            else:
                parts.append((lo, hi))
        low, high = polygon[:, 0].min(), polygon[:, 0].max()
        threshold = max(self.minimum_width, self.minimum_ratio * (high - low))
        result = []
        for lo, hi in parts:
            rounding = 4 * np.finfo(float).eps * max(abs(lo), abs(hi), threshold)
            if hi - lo < threshold - rounding:
                continue
            if lo == low and hi == high:
                result.append(candidate)
            else:
                clipped = _clip(_clip(polygon, lo, True), hi, False) @ FRAME
                result.append(_fit(clipped, candidate))
        return tuple(result)


def corner_error(first: Box, second: Box) -> float:
    """Squared RMS corner displacement in m², invariant to equivalent OBB axes."""
    a, b = _footprint(first), _footprint(second)
    return min(float(np.mean(np.sum((a - np.roll(b, shift, axis=0))**2, axis=1)))
               for shift in range(4))
