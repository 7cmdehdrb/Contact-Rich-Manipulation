"""Fixed-command clutter geometry, checked by the original independent verifier."""

from dataclasses import replace
import importlib.util
import math
import sys

import numpy as np
import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT, _module_from_file


@pytest.fixture(scope="module")
def layout():
    return _module_from_file(
        "_baseline_cpu_clutter_layout", MODULE_ROOT / "contact_sweep_clutter/layout.py",
    )


@pytest.fixture(scope="module")
def validate_scenes():
    root = PROJECT_ROOT / "src/scene_randomization"
    name = "_baseline_cpu_scene_tool"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.validate_scenes


def baseline_commands(layout, goal_events, count, seed):
    """Use production baseline samplers, including ray shortening and cones."""
    torch.manual_seed(seed)
    target_xy = goal_events._sample_uniform_disk(count, 0.02, "cpu")
    direction = goal_events._sample_planar_unit_vectors(count, "cpu", math.radians(15.0))
    distance = goal_events._goal_ray_distance(target_xy, direction, 0.15, (0.14, 0.44))
    goal_xy = target_xy + direction * distance[:, None]
    return (
        layout.shelf_xy_to_generator_cm(target_xy.numpy()),
        layout.shelf_xy_to_generator_cm(goal_xy.numpy()),
    )


def test_coordinate_transform_preserves_front_rotation_height_convention(layout):
    # Front/rear centre and the side corners, in the existing shelf local XY.
    shelf = np.array([[0.2, 0.0], [-0.2, 0.0], [0.2, -0.5], [-0.2, 0.5]])
    expected = np.array([[0.0, 0.0], [0.0, 40.0], [-50.0, 0.0], [50.0, 40.0]])
    np.testing.assert_allclose(layout.shelf_xy_to_generator_cm(shelf), expected, atol=1e-14)
    np.testing.assert_allclose(layout.generator_cm_to_shelf_xy(expected), shelf, atol=1e-14)
    origin = layout.generator_cm_to_shelf_xy([[0.0, 20.0]])
    front = layout.generator_cm_to_shelf_xy([[0.0, 19.0]])
    np.testing.assert_allclose(front - origin, [[0.01, 0.0]], atol=1e-14)
    # +generator X -> +shelf Y, +generator Y -> -shelf X: determinant +1.
    basis = layout.generator_cm_to_shelf_xy([[1.0, 20.0], [0.0, 21.0]]) - origin
    assert np.linalg.det(basis.T) == pytest.approx(1e-4)


def test_baseline_goals_and_targets_remain_fixed_and_all_geometry_passes(layout, validate_scenes, load_mdp):
    target, goal = baseline_commands(layout, load_mdp("goal_events"), 4096, seed=20261020)
    before_target, before_goal = target.copy(), goal.copy()
    batch = layout.ClutterLayoutGenerator(seed=19).generate_batch(target, goal)
    np.testing.assert_array_equal(target, before_target)
    np.testing.assert_array_equal(goal, before_goal)
    np.testing.assert_array_equal(batch.target, before_target)
    np.testing.assert_array_equal(batch.goal, before_goal)
    assert batch.obstacles.shape == (4096, 3, 2)
    assert batch.config.gray_obstacles == 1
    assert batch.config.rear_obstacles == 1
    assert 2 * batch.config.radius + batch.config.red_margin == pytest.approx(10.3)
    # No change to the standalone generator defaults.
    assert layout.Config().obstacles == 6
    assert layout.Config().red_margin == 0.3
    report = validate_scenes(batch)
    assert report["valid"], report
    assert report["minima"]["surface_gap_cm"] >= 1.6 - 1e-8
    assert report["minima"]["wrist_projection_clearance_cm"] >= 2.3 - 1e-8
    assert report["minima"]["approach_obstacle_clearance_cm"] >= 0.1 - 1e-8
    # Returned positions own their storage, so downstream buffers cannot alter inputs.
    batch.target[0, 0] += 1.0
    batch.goal[0, 0] += 1.0
    np.testing.assert_array_equal(target, before_target)
    np.testing.assert_array_equal(goal, before_goal)


def test_geometry_rng_reproducible_and_continues_between_resets(layout):
    target = np.tile([0.0, 20.0], (9, 1))
    goal = np.tile([15.0, 20.0], (9, 1))
    left, right = layout.ClutterLayoutGenerator(seed=42), layout.ClutterLayoutGenerator(seed=42)
    first = left.generate(target, goal)
    np.testing.assert_array_equal(first, right.generate(target, goal))
    second = left.generate(target, goal)
    np.testing.assert_array_equal(second, right.generate(target, goal))
    assert not np.array_equal(first, second)


def test_layout_failure_keeps_supplied_command_and_is_explicit(layout):
    config = replace(
        layout.Config(), max_neighbors=0, density_radius=200.0,
        scene_attempts=3, placement_rounds=1,
    )
    target = np.array([[0.0, 20.0], [1.0, 20.0]])
    goal = np.array([[15.0, 20.0], [-14.0, 20.0]])
    before_target, before_goal = target.copy(), goal.copy()
    with pytest.raises(layout.GenerationError, match="commands were NOT replaced") as caught:
        layout.ClutterLayoutGenerator(seed=42, config=config).generate(target, goal)
    np.testing.assert_array_equal(caught.value.indices, [0, 1])
    np.testing.assert_allclose(caught.value.angle, [0.0, np.pi], atol=1e-14)
    np.testing.assert_array_equal(caught.value.distance, [15.0, 15.0])
    np.testing.assert_array_equal(target, before_target)
    np.testing.assert_array_equal(goal, before_goal)
    assert caught.value.reasons["rear"] == 6


@pytest.mark.parametrize(("target", "goal", "message"), [
    ([[0.0, 20.0, 1.0]], [[15.0, 20.0, 1.0]], "shapes"),
    ([[0.0, 20.0]], [[float("nan"), 20.0]], "finite"),
    ([[40.0, 20.0]], [[25.0, 20.0]], "safe target"),
    ([[0.0, 20.0]], [[46.0, 20.0]], "object bounds"),
    ([[0.0, 20.0]], [[0.0, 20.0]], "distance"),
])
def test_invalid_fixed_inputs_are_rejected_not_repaired(layout, target, goal, message):
    with pytest.raises(ValueError, match=message):
        layout.ClutterLayoutGenerator().generate(target, goal)


def test_empty_reset_batch_and_maximum_observation_capacity(layout):
    assert layout.ClutterLayoutGenerator().generate(np.empty((0, 2)), np.empty((0, 2))).shape == (0, 3, 2)
    with pytest.raises(ValueError, match="at most 10"):
        layout.ClutterLayoutGenerator(config=replace(layout.Config(), obstacles=11))
