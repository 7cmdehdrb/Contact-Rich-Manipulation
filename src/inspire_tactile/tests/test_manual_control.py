import math
import tempfile
import unittest
from pathlib import Path

from inspire_tactile.manual_control import JointControls
from inspire_tactile.model import build_model, repo_root


class ManualControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        _, cls.model = build_model(repo_root(), Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.control = JointControls(self.model)

    def test_individual_follower_does_not_move_parent(self):
        self.assertEqual(len(self.control.names), 12)
        before = dict(self.control.desired)
        self.control.set_target("inspire_left_index_2_joint", 0.5)
        changed = [n for n in before if before[n] != self.control.desired[n]]
        self.assertEqual(changed, ["inspire_left_index_2_joint"])

    def test_limits_and_invalid_input(self):
        name = "inspire_left_thumb_3_joint"
        self.control.set_target(name, 200)
        self.assertEqual(self.control.desired[name], 0.5)
        self.control.set_target(name, -1)
        self.assertEqual(self.control.desired[name], 0)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.control.set_target(name, value)
        with self.assertRaises(ValueError):
            self.control.set_target("shoulder_pan_joint", 0.1)

    def test_linked_nested_mimics_and_toggle(self):
        self.control.set_linked(True)
        self.control.set_target("inspire_left_thumb_2_joint", 0.4)
        self.assertAlmostEqual(self.control.desired["inspire_left_thumb_4_joint"], 0.4 * 0.8024 * 0.9487)
        with self.assertRaises(ValueError):
            self.control.set_target("inspire_left_thumb_3_joint", 0.1)
        self.control.set_linked(False)
        self.control.set_target("inspire_left_thumb_3_joint", 0.1)
        self.assertAlmostEqual(self.control.desired["inspire_left_thumb_2_joint"], 0.4)

    def test_slew_limit_and_reset(self):
        self.control.preset("close")
        before = dict(self.control.commanded)
        result = self.control.step(0.1)
        for name in result:
            self.assertLessEqual(abs(result[name] - before[name]), 0.06000001)
        self.control.reset_requested = True
        self.control.reset()
        self.assertFalse(self.control.reset_requested)
        self.assertEqual(self.control.commanded, self.model["open_targets"])
        with self.assertRaises(ValueError):
            self.control.step(float("nan"))

    def test_hold_and_presets_are_bounded(self):
        for preset in ("close", "open"):
            self.control.preset(preset)
            for name, value in self.control.desired.items():
                self.assertTrue(math.isfinite(value))
                self.assertLessEqual(value, self.control.specs[name]["upper"])
        actual = {n: 0.1 for n in self.control.names}
        self.control.hold(actual)
        self.assertEqual(self.control.desired, actual)


if __name__ == "__main__":
    unittest.main()
