"""Separate Sweep-family entry point using the shared, unchanged Sweep lifecycle."""
from .stabilize_sweep import SweepOnlyEnv


class SweepEnv(SweepOnlyEnv):
    """Sweep v0; reset, contact sampling and telemetry use the shared runtime."""
