"""Playback-only lifecycle extension; registered Fusion configs stay unchanged."""

from .env import ContactSweepClutterFusionEnv
from .telemetry import PlaybackTelemetry


class ContactSweepClutterFusionTelemetryEnv(ContactSweepClutterFusionEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.playback_telemetry = None
        self._telemetry_in_step = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.playback_telemetry = PlaybackTelemetry(self)

    def _reset_idx(self, env_ids):
        telemetry = self.playback_telemetry
        if telemetry is not None and self._telemetry_in_step:
            telemetry.capture(self, env_ids)
        super()._reset_idx(env_ids)
        if telemetry is not None:
            telemetry.reset(env_ids)

    def step(self, action):
        self.playback_telemetry.begin_step()
        self._telemetry_in_step = True
        try:
            result = super().step(action)
        finally:
            self._telemetry_in_step = False
        self.playback_telemetry.capture(self, self.sensor_data_fresh.nonzero().flatten())
        self.playback_telemetry.finish_step()
        return result
