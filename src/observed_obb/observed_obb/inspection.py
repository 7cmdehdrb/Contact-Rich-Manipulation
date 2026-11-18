"""Explicit snapshot lifecycle. Viewer frames never call observation generation."""
from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING

from .observation import Box, Observation, observe

if TYPE_CHECKING:
    from .display import ObservationDisplay


class Inspection:
    def __init__(self, new_layout: Callable[[], tuple[Box, ...]], display: "ObservationDisplay"):
        self._new_layout = new_layout
        self.display = display
        self.boxes: tuple[Box, ...] = ()
        self.snapshot: tuple[Observation, ...] = ()
        self.reference_visible = False
        self.observation_seconds = 0.0
        self.display_seconds = 0.0

    def new_placement(self) -> None:
        self.boxes = tuple(self._new_layout())
        self.reobserve()

    def reobserve(self) -> None:
        start = perf_counter()
        snapshot = observe(self.boxes)
        self.observation_seconds = perf_counter() - start
        start = perf_counter()
        self.display.show(snapshot)
        self.display.reference(self.boxes, self.reference_visible)
        self.display_seconds = perf_counter() - start
        self.snapshot = snapshot

    def toggle_reference(self) -> None:
        self.reference_visible = not self.reference_visible
        self.display.set_reference_visible(self.reference_visible)
