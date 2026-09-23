"""Version 1 wire contract; no Isaac Sim or third-party dependencies."""

import json
import math

NAMES = ["inspire_palm_force_sensor"] + [
    f"inspire_{finger}_force_sensor_{i}"
    for finger, count in (("thumb", 4), ("index", 3), ("middle", 3), ("ring", 3), ("little", 3))
    for i in range(1, count + 1)
]


def decode(data):
    packet = json.loads(data)
    if not isinstance(packet, dict) or packet.get("version") != 1 or packet.get("names") != NAMES:
        raise ValueError("Unsupported tactile schema or channel ordering")
    if packet.get("frame_id") != "world" or packet.get("position_kind") != "sensor_link_origin":
        raise ValueError("Unsupported position frame/semantics")
    for key in ("sequence", "env_id"):
        if type(packet.get(key)) is not int or packet[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not isinstance(packet.get("session"), str) or len(packet["session"]) != 32:
        raise ValueError("Invalid session")
    if packet.get("phase") not in ("open", "close", "hold", "release", "released", "manual"):
        raise ValueError("Invalid phase")
    for key in ("time_s", "threshold_N"):
        value = packet.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if packet["threshold_N"] == 0:
        raise ValueError("Threshold must be positive")
    if packet.get("force_source", "sensor_body_raw") not in ("sensor_body_raw", "contact_region_projection"):
        raise ValueError("Unsupported force source")
    tolerance = packet.get("region_tolerance_m", 0.0)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid region tolerance")
    for key, vector in (("positions_w_m", True), ("forces_w_N", True), ("norms_N", False)):
        rows = packet.get(key)
        if not isinstance(rows, list) or len(rows) != 17:
            raise ValueError(f"Invalid {key} shape")
        for row in rows:
            values = row if vector else [row]
            if not isinstance(values, list) or (vector and len(values) != 3):
                raise ValueError(f"Invalid {key} vector")
            if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
                raise ValueError(f"Non-finite {key}")
    active = packet.get("active")
    if not isinstance(active, list) or len(active) != 17 or any(type(x) is not int or x not in (0, 1) for x in active):
        raise ValueError("Invalid binary array")
    if any(x < 0 for x in packet["norms_N"]):
        raise ValueError("Negative force norm")
    return packet
