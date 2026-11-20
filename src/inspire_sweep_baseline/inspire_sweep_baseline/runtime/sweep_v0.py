"""Separate Sweep-family entry point using the shared, unchanged Sweep lifecycle."""
from .stabilize_sweep import SweepOnlyEnv
from ..mdp.common.sweep_v0_state import SingleContactSweepState


class SweepEnv(SweepOnlyEnv):
    """Sweep v0; reset, contact sampling and telemetry use the shared runtime."""


class SweepSingleContactEnv(SweepEnv):
    """Keep the Sweep lifecycle; replace only accumulated contact credit."""

    _state_type = SingleContactSweepState
