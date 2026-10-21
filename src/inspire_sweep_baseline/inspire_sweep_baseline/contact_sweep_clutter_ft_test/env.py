"""Capture terminal transitions before reset without extra observation calls."""

from ..contact_sweep_clutter_v1_curriculum.env import ContactSweepClutterV1CurriculumEnv
from .telemetry import WrenchTelemetry


class ContactSweepClutterFTTestEnv(ContactSweepClutterV1CurriculumEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.ft_telemetry = None
        self._ft_in_step = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.ft_telemetry = WrenchTelemetry(self)

    def _reset_idx(self, env_ids):
        telemetry = self.ft_telemetry
        if telemetry is not None and self._ft_in_step:
            # Reward/termination managers have finished; physical state and
            # flags still belong to the episode that is about to end.
            telemetry.capture(self, env_ids)
        super()._reset_idx(env_ids)
        if telemetry is not None:
            telemetry.reset(env_ids)

    def step(self, action):
        self.ft_telemetry.begin_step()
        self._ft_in_step = True
        try:
            result = super().step(action)
        finally:
            self._ft_in_step = False
        # Reset rows have already been captured above. Their fresh flag is
        # false now, so they cannot overwrite terminal samples with reset zeros.
        self.ft_telemetry.capture(self, self.sensor_data_fresh.nonzero().flatten())
        self.ft_telemetry.finish_step()
        return result
