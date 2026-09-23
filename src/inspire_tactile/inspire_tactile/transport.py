"""ROS-independent, bounded localhost transport to the Python 3.10 ROS node."""

import json
import socket
import uuid
import warnings


class TactileSender:
    def __init__(self, port=9870):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.address = ("127.0.0.1", port)
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.dropped = 0

    def send(self, *, time_s, phase, names, threshold, positions, forces, norms, bits,
             force_source="sensor_body_raw", region_tolerance=0.0):
        # One small datagram per environment, not a potentially oversized batch.
        for env_id in range(len(bits)):
            packet = {
                "version": 1, "session": self.session, "sequence": self.sequence,
                "time_s": time_s, "phase": phase, "frame_id": "world", "env_id": env_id,
                "names": list(names), "threshold_N": threshold,
                "position_kind": "sensor_link_origin", "positions_w_m": positions[env_id],
                "forces_w_N": forces[env_id], "norms_N": norms[env_id], "active": bits[env_id],
                "force_source": force_source,
                "region_tolerance_m": region_tolerance if force_source == "contact_region_projection" else 0.0,
            }
            data = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode()
            if len(data) > 60000:
                raise ValueError("Tactile datagram exceeds safe UDP payload size")
            try:
                self.socket.sendto(data, self.address)
            except OSError as exc:
                self.dropped += 1
                if self.dropped == 1:
                    warnings.warn(f"Tactile UDP sample dropped: {exc}", stacklevel=2)
        self.sequence += 1

    def close(self):
        self.socket.close()
