"""Bounded, rate-limited hand commands; independent of Isaac Sim and the GUI."""

import math
import xml.etree.ElementTree as ET


class JointControls:
    def __init__(self, model, speed=0.6, linked=False):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Target speed must be finite and positive [rad/s]")
        self.speed = speed
        self.specs = {}
        hand = ET.parse(model["source_hand"]).getroot()
        for joint in hand.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            mimic = joint.find("mimic")
            self.specs["inspire_" + joint.get("name")] = {
                "lower": float(limit.get("lower")), "upper": float(limit.get("upper")),
                "velocity": float(limit.get("velocity")),
                "mimic": None if mimic is None else (
                    "inspire_" + mimic.get("joint"), float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0"))),
            }
        self.names = sorted(self.specs, key=lambda n: (
            ("thumb", "index", "middle", "ring", "little").index(n.split("_")[2]), n))
        self.presets = {"open": dict(model["open_targets"]), "close": dict(model["closed_targets"])}
        self.desired = dict(self.presets["open"])
        self.commanded = dict(self.desired)
        self.linked = linked
        self.reset_requested = False
        self._link(self.desired)

    def _link(self, values):
        if not self.linked:
            return
        pending = {n for n in self.names if self.specs[n]["mimic"]}
        while pending:
            resolved = []
            for name in pending:
                parent, multiplier, offset = self.specs[name]["mimic"]
                if parent in pending:
                    continue
                values[name] = self._clamp(name, values[parent] * multiplier + offset)
                resolved.append(name)
            if not resolved:
                raise ValueError("Cyclic mimic graph")
            pending.difference_update(resolved)

    def _clamp(self, name, value):
        if not math.isfinite(value):
            raise ValueError("Joint target must be finite")
        spec = self.specs[name]
        return min(spec["upper"], max(spec["lower"], value))

    def set_target(self, name, radians):
        if name not in self.specs:
            raise ValueError(f"Unknown hand joint: {name}")
        if self.linked and self.specs[name]["mimic"]:
            raise ValueError("Mimic follower is read-only in linked mode")
        self.desired[name] = self._clamp(name, radians)
        self._link(self.desired)

    def set_linked(self, enabled):
        self.linked = bool(enabled)
        self._link(self.desired)

    def preset(self, name):
        self.desired = dict(self.presets[name])
        self._link(self.desired)

    def hold(self, actual):
        self.desired = {n: self._clamp(n, actual[n]) for n in self.names}
        self._link(self.desired)

    def reset(self):
        self.preset("open")
        self.commanded = dict(self.desired)
        self.reset_requested = False

    def step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Each commanded target obeys its own velocity cap, including followers.
        # While transitioning between independent/linked modes, follower ratios
        # may temporarily differ; they converge to the linked desired targets.
        for name in self.names:
            delta = self.desired[name] - self.commanded[name]
            maximum = min(self.speed, self.specs[name]["velocity"]) * dt
            self.commanded[name] += min(maximum, max(-maximum, delta))
        return dict(self.commanded)
