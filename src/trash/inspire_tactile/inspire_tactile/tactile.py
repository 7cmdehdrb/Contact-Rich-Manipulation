"""A stable 17-channel tensor interface, independent of Isaac Sim imports."""

from __future__ import annotations

import math
import torch


def binary_contact(forces_w: torch.Tensor, threshold: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., sensors, 3) normal-contact force in N -> magnitudes and uint8 bits.

    No smoothing or hysteresis: equality with the threshold MUST produce one.
    Non-finite input is an error, never silently classified as no contact.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if forces_w.ndim < 2 or forces_w.shape[-1] != 3:
        raise ValueError("Expected (..., sensors, 3) force tensor")
    if not torch.isfinite(forces_w).all():
        raise RuntimeError("Non-finite contact forces; check simulation stability")
    magnitude = torch.linalg.vector_norm(forces_w, ord=2, dim=-1)
    return magnitude, (magnitude >= threshold).to(torch.uint8)


class TactileReader:
    """Resolve channel order by name, never by PhysX's incidental body ordering."""

    force_source = "sensor_body_raw"

    def __init__(self, sensor, names: list[str], threshold: float = 0.01):
        if set(sensor.body_names) != set(names) or len(sensor.body_names) != len(names):
            raise RuntimeError(f"Expected {names}; ContactSensor found {sensor.body_names}")
        self.sensor = sensor
        self.names = names
        self.indices = [sensor.body_names.index(name) for name in names]
        self.threshold = threshold

    def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forces = self.sensor.data.net_forces_w[:, self.indices, :]
        self.raw_forces_w = forces
        magnitude, bits = binary_contact(forces, self.threshold)
        return forces, magnitude, bits

    def diagnostics(self):
        return {"force_source": self.force_source,
                "raw_sensor_force_w_N": self.raw_forces_w.cpu().tolist()}
