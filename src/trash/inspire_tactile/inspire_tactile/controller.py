"""Deterministic close/hold/open schedule; no training and no object-pose feedback."""

from __future__ import annotations

PHASES = (("open", 1.5), ("close", 3.0), ("hold", 2.0), ("release", 3.0), ("released", 1.5))
CYCLE_SECONDS = sum(duration for _, duration in PHASES)


def phase_at(t: float) -> tuple[str, float]:
    """Return phase and closure fraction, with smooth zero-velocity endpoints."""
    t = max(t, 0.0) % CYCLE_SECONDS
    for name, duration in PHASES:
        if t < duration:
            s = t / duration
            smooth = s * s * (3.0 - 2.0 * s)
            return name, {"open": 0.0, "close": smooth, "hold": 1.0,
                          "release": 1.0 - smooth, "released": 0.0}[name]
        t -= duration
    return "open", 0.0
