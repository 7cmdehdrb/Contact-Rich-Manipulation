"""Sweep-family contact accounting variants."""
import torch
from .stabilize_sweep_state import SweepOnlyState


class SingleContactSweepState(SweepOnlyState):
    """Two-of-three recognition gives unit contact credit without a ramp."""

    def _init_contact_factor(self, count, device):
        self.contact_recognized = torch.zeros(count, device=device, dtype=torch.bool)

    def _reset_contact_factor(self, ids):
        self.contact_recognized[ids] = False

    def _contact_factor(self, recognized, normalize_contact_count):
        self.contact_recognized.copy_(recognized)
        return self.contact_recognized.float()
