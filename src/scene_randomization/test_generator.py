"""Simulator-free contract, distribution, failure and certificate tests."""

from dataclasses import replace
import json
import tempfile
import unittest

import numpy as np

from .generator import (Config, GenerationError, SceneBatch, SceneGenerator, generate_parallel,
                        downward_sweep_distance2, zone_masks)
from .validation import _downward_distances, distribution_report, validate_scenes


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = SceneGenerator(seed=17).generate(4096)

    def test_all_geometry_contracts(self):
        report = validate_scenes(self.batch)
        self.assertTrue(report["valid"], report)
        self.assertGreaterEqual(report["minima"]["surface_gap_cm"], 1.6 - 1e-8)
        self.assertGreaterEqual(report["minima"]["approach_obstacle_clearance_cm"], 0.1 - 1e-8)
        self.assertGreaterEqual(report["minima"]["wrist_projection_clearance_cm"], 0.3 - 1e-8)

    def test_horizontal_red_reaches_front_and_gray_flanks_it(self):
        points = np.array([[[10., 5.], [-11., 5.], [-16., 5.], [10., 29.], [-8., 24.]]])
        red, gray, _ = zone_masks(points, np.array([[0., 20.]]), np.array([[1., 0.]]),
                                  np.array([20.]), Config())
        np.testing.assert_array_equal(red[0], [True, False, False, False, False])
        np.testing.assert_array_equal(gray[0], [False, True, False, True, True])

    def test_diagonal_upper_boundary_is_not_a_bounding_rectangle(self):
        d = np.array([[1., 1.]]) / np.sqrt(2)
        q = np.array([[[0., 34.], [14., 34.], [0., 5.]]])
        red, _, _ = zone_masks(q, np.array([[0., 20.]]), d, np.array([20.]), Config())
        np.testing.assert_array_equal(red[0], [False, True, True])

    def test_projection_is_always_world_minus_y(self):
        for d in ([0., 1.], [0., -1.], [1., 0.], [-1., 0.]):
            with self.subTest(direction=d):
                q = np.array([[[0., 5.], [0., 40.]]])
                red, _, _ = zone_masks(q, np.array([[0., 20.]]), np.array([d]), np.array([5.]), Config())
                np.testing.assert_array_equal(red[0], [True, False])

    def test_projection_against_independent_geometry(self):
        rng = np.random.default_rng(3)
        angles = np.r_[np.arange(4) * np.pi / 2, rng.uniform(0, 2 * np.pi, 30)]
        directions = np.stack((np.cos(angles), np.sin(angles)), -1)
        directions[1] = [0., 1.]  # exact vertical, not floating-point cos(pi/2)
        directions[3] = [0., -1.]
        p = rng.uniform([-30., 12.], [30., 28.], (len(angles), 2))
        distance = rng.uniform(5, 20, len(angles))
        q = rng.uniform([-50., -50.], [50., 60.], (len(angles), 200, 2))
        goal = p + distance[:, None] * directions
        projected = downward_sweep_distance2(q, p, directions, distance)
        independent = _downward_distances(q, p, goal) ** 2
        np.testing.assert_allclose(projected, independent, atol=1e-9)
        reversed_projection = downward_sweep_distance2(q, goal, -directions, distance)
        np.testing.assert_allclose(projected, reversed_projection, atol=1e-9)
        lower = q - [0, 30]
        self.assertTrue(np.all(downward_sweep_distance2(lower, p, directions, distance) <= projected + 1e-9))

    def test_validator_detects_wrist_conflict_outside_old_capsule(self):
        batch = SceneGenerator(seed=5).generate(1)
        batch.target[0], batch.goal[0] = [0, 20], [20, 20]
        batch.angle[0], batch.distance[0] = 0, 20
        batch.obstacles[0, -1] = [10, 5]  # 15 cm from S; directly under its x projection
        report = validate_scenes(batch)
        self.assertEqual(report["failures"]["wrist_projection_obstacle"], 1)

    def test_gray_proposals_sample_extended_front_region(self):
        # A capsule-only proposal misses these physically relevant Wrist neighbors.
        p, goal = self.batch.target, self.batch.goal
        gray = self.batch.obstacles[:, :self.batch.config.gray_obstacles]
        below_both = gray[..., 1] < np.minimum(p[:, None, 1], goal[:, None, 1]) - self.batch.config.gray_radius
        rel, direction = gray - p[:, None], self.batch.direction
        xi = np.einsum("bpi,bi->bp", rel, direction)
        eta = rel[..., 1] * direction[:, None, 0] - rel[..., 0] * direction[:, None, 1]
        old_rear = ((xi >= -self.batch.config.rear_length) & (xi <= 0)
                    & (np.abs(eta) <= self.batch.config.rear_width / 2))
        self.assertGreater(int((below_both & ~old_rear).sum()), 0)

    def test_joint_coverage_and_continuous_jitter(self):
        report = distribution_report(self.batch)
        self.assertLessEqual(report["joint_count_max"] - report["joint_count_min"], 1)
        for harmonic in report["angle_fourier_magnitudes_1_to_4"]:
            self.assertLess(harmonic, 0.025)
        self.assertAlmostEqual(report["distance_mean_cm"], 12.5, delta=0.08)
        self.assertAlmostEqual(report["normalized_distance_variance"], 1 / 12, delta=0.003)
        self.assertAlmostEqual(report["angle_within_bin_mean"], 0.5, delta=0.025)
        self.assertAlmostEqual(report["distance_within_bin_mean"], 0.5, delta=0.025)

    def test_schedule_persists_across_calls(self):
        generator = SceneGenerator(seed=42)
        commands = [generator.commands(n) for n in (1, 17, 100, 311, 147)]  # 3 full cycles
        angle, distance = (np.concatenate([row[k] for row in commands]) for k in (0, 1))
        counts, _, _ = np.histogram2d(angle, distance, bins=(24, 8), range=((0, 2 * np.pi), (5, 20)))
        np.testing.assert_array_equal(counts, np.full((24, 8), 3))

    def test_seed_is_reproducible(self):
        a, b = SceneGenerator(seed=99).generate(100), SceneGenerator(seed=99).generate(100)
        for key in ("target", "goal", "angle", "distance", "obstacles", "approach", "attempts"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))

    def test_resampling_keeps_commands(self):
        # Dense settings force actual layout retries; commands must stay identical
        # to those produced before any scene construction or retry takes place.
        c = replace(Config(), obstacles=10, gray_obstacles=4, rear_obstacles=2,
                    max_per_cell=2, max_neighbors=3, candidates=8, placement_rounds=4,
                    scene_attempts=100)
        expected_angle, expected_distance = SceneGenerator(c, seed=8).commands(192)
        batch = SceneGenerator(c, seed=8).generate(192)
        np.testing.assert_array_equal(batch.angle, expected_angle)
        np.testing.assert_array_equal(batch.distance, expected_distance)
        self.assertTrue(np.any(batch.attempts > 1))
        self.assertTrue(validate_scenes(batch)["valid"])

    def test_parallel_preserves_global_commands(self):
        a = generate_parallel(600, seed=81, workers=1)
        b = generate_parallel(600, seed=81, workers=2)
        np.testing.assert_array_equal(a.angle, b.angle)
        np.testing.assert_array_equal(a.distance, b.distance)
        self.assertTrue(validate_scenes(b)["valid"])

    def test_parallel_failure_is_reported(self):
        c = replace(Config(), max_neighbors=0, density_radius=200, scene_attempts=1, placement_rounds=1)
        expected_angle, expected_distance = SceneGenerator(c, seed=9).commands(2)
        with self.assertRaises(GenerationError) as caught:
            generate_parallel(2, c, seed=9, workers=2)
        exc = caught.exception
        np.testing.assert_array_equal(exc.indices, [0, 1])
        np.testing.assert_array_equal(exc.angle, expected_angle)
        np.testing.assert_array_equal(exc.distance, expected_distance)

    def test_validator_detects_corrupted_scene(self):
        batch = SceneGenerator(seed=5).generate(1)
        batch.obstacles[0, 0] = batch.target[0]
        report = validate_scenes(batch)
        self.assertFalse(report["valid"])
        self.assertEqual(report["failures"]["minimum_gap"], 1)
        self.assertEqual(report["failures"]["red_obstacle"], 1)

    def test_validator_checks_continuous_path(self):
        batch = SceneGenerator(seed=5).generate(1)
        # Endpoints alone clear this obstacle, but the long front-entry segment
        # passes through its center. The verifier must examine the full segment.
        batch.obstacles[0, -1] = (batch.approach[0, 0] + batch.approach[0, 1]) / 2
        report = validate_scenes(batch)
        self.assertEqual(report["failures"]["approach_obstacle"], 1)

    def test_density_includes_existing_objects(self):
        c = replace(Config(), obstacles=3, gray_obstacles=3, rear_obstacles=0,
                    density_radius=14, max_neighbors=2)
        batch = SceneGenerator(c, seed=4).generate(1)
        batch.target[0] = [0, 20]
        batch.obstacles[0] = [[-11, 20], [11, 20], [0, 31]]
        # All obstacles have only one neighbor; the target has three.
        self.assertEqual(validate_scenes(batch)["failures"]["local_density"], 1)

    def test_nonfinite_goal_detected(self):
        batch = SceneGenerator(seed=5).generate(1)
        batch.goal[0] = np.nan
        self.assertEqual(validate_scenes(batch)["failures"]["nonfinite"], 1)

    def test_npz_roundtrip(self):
        batch = SceneGenerator(seed=0).generate(2)
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/scenes.npz"
            batch.save(path)
            loaded = SceneBatch.load(path)
        np.testing.assert_array_equal(batch.obstacles, loaded.obstacles)
        self.assertTrue(validate_scenes(loaded)["valid"])

    def test_legacy_npz_is_not_silently_reclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/legacy.npz"
            config = {"radius": 4.0}  # no zone_model: old capsule-only schema
            np.savez(path, config_json=np.array(json.dumps(config)))
            with self.assertRaisesRegex(ValueError, "Legacy scenes"):
                SceneBatch.load(path)

    def test_empty_obstacle_mode(self):
        c = replace(Config(), obstacles=0, gray_obstacles=0, rear_obstacles=0)
        batch = SceneGenerator(c, seed=0).generate(192)
        self.assertEqual(batch.obstacles.shape, (192, 0, 2))
        self.assertTrue(validate_scenes(batch)["valid"])

    def test_invalid_or_biased_support_rejected(self):
        for c in (replace(Config(), distance_max=30), replace(Config(), contact_clearance=0),
                  replace(Config(), rear_obstacles=4), replace(Config(), candidates=0),
                  replace(Config(), radius=-1)):
            with self.subTest(config=c), self.assertRaises(ValueError):
                SceneGenerator(c)
        with self.assertRaises(ValueError):
            SceneGenerator().generate(0)


if __name__ == "__main__":
    unittest.main()
