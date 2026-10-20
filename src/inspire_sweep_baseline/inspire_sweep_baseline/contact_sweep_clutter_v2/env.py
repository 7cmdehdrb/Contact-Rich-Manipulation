"""Keep the v1 lifecycle while replacing the wrist measurement pipeline."""

from ..contact_sweep_clutter_v1.env import ContactSweepClutterV1Env


class ContactSweepClutterV2Env(ContactSweepClutterV1Env):
    """Sensor and reward managers own compensation and episode-local history."""
