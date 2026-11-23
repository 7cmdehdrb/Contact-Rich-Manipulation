"""Sweep-family goal restrictions using the unchanged circle placement pipeline."""
import math

from .wrench_refactor_layout import CircleResetLayout


class SingleContactZone3ResetLayout(CircleResetLayout):
    """Only sample pull goals: shelf-local +X (6 o'clock), within +/-30 degrees."""

    def sample_commands(self, ids, generator, **command_params):
        # The shared sampler uses atan2(Y, X), not the region clock angle.
        # Preserve target draws, distance/edge rules and obstacle placement.
        command_params['goal_angle_range'] = (-math.pi / 6, math.pi / 6)
        return super().sample_commands(ids, generator, **command_params)
