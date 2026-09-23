import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

from inspire_tactile_ros.protocol import NAMES, decode


def sample():
    return {"version": 1, "session": "a" * 32, "sequence": 7, "env_id": 0,
            "phase": "hold", "time_s": 5.0, "frame_id": "world", "names": NAMES,
            "position_kind": "sensor_link_origin", "threshold_N": 0.01,
            "positions_w_m": [[0.1, 0.2, 0.3] for _ in NAMES],
            "forces_w_N": [[0.01, 0.0, 0.0] for _ in NAMES],
            "norms_N": [0.01] * 17, "active": [1] * 17}


class ProtocolTest(unittest.TestCase):
    def test_sim_sender_contract_and_multi_environment(self):
        # Pure-stdlib sender can be tested in ROS Python without importing Sim.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inspire_tactile"))
        from inspire_tactile.transport import TactileSender
        packet = sample()
        with patch("inspire_tactile.transport.socket.socket") as sock:
            sender = TactileSender(9871)
            for _ in range(2):
                sender.send(time_s=packet["time_s"], phase=packet["phase"], names=NAMES,
                            threshold=packet["threshold_N"], positions=[packet["positions_w_m"]] * 2,
                            forces=[packet["forces_w_N"]] * 2, norms=[packet["norms_N"]] * 2,
                            bits=[packet["active"]] * 2,
                            force_source="contact_region_projection", region_tolerance=.002)
            calls = sock.return_value.sendto.call_args_list
            self.assertEqual(len(calls), 4)
            decoded = [decode(call.args[0]) for call in calls]
            self.assertEqual([p["env_id"] for p in decoded], [0, 1, 0, 1])
            self.assertEqual([p["sequence"] for p in decoded], [0, 0, 1, 1])
            self.assertEqual(decoded[0]["active"], packet["active"])
            self.assertEqual(decoded[0]["force_source"], "contact_region_projection")
            self.assertEqual(decoded[0]["region_tolerance_m"], .002)
            self.assertEqual(calls[0].args[1], ("127.0.0.1", 9871))
            sender.close()
            sock.return_value.close.assert_called_once()

    def test_round_trip(self):
        self.assertEqual(decode(json.dumps(sample())), sample())

    def test_manual_phase_is_accepted(self):
        packet = sample()
        packet["phase"] = "manual"
        self.assertEqual(decode(json.dumps(packet))["phase"], "manual")

    def test_projection_metadata_is_preserved_and_validated(self):
        packet = sample()
        packet.update(force_source="contact_region_projection", region_tolerance_m=.002)
        self.assertEqual(decode(json.dumps(packet)), packet)
        for key, value in (("force_source", "unknown"), ("region_tolerance_m", -1),
                           ("region_tolerance_m", float("nan"))):
            invalid = dict(packet)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(json.dumps(invalid))

    def test_reject_bad_shape_order_nan_binary_frame(self):
        for field, value in (("names", NAMES[::-1]), ("active", [0] * 16),
                             ("active", [2] * 17), ("norms_N", [float("nan")] * 17),
                             ("positions_w_m", [[1, 2]] * 17), ("frame_id", "hand"),
                             ("sequence", -1), ("threshold_N", 0), ("phase", "unknown")):
            packet = copy.deepcopy(sample())
            packet[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode(json.dumps(packet))

    def test_markers_use_real_positions_and_expire(self):
        from builtin_interfaces.msg import Time
        from inspire_tactile_ros.bridge import make_markers
        packet = sample()
        packet["active"][0] = 0
        result = make_markers(packet, Time(sec=12))
        self.assertEqual(len(result.markers), 34)
        self.assertEqual(result.markers[0].pose.position.x, 0.1)
        self.assertEqual(result.markers[0].header.frame_id, "world")
        self.assertEqual(result.markers[0].lifetime.sec, 1)
        self.assertAlmostEqual(result.markers[0].color.g, 0.45)
        self.assertAlmostEqual(result.markers[2].color.g, 0.9)


if __name__ == "__main__":
    unittest.main()
