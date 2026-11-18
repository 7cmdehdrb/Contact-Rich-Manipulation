"""Contract tests: placed boxes -> independent observed OBB snapshot."""
import math

import numpy as np
import pytest

from observed_obb import Box, observe


def test_fully_visible_box_preserves_geometry():
    box = Box(center=(0.3, -0.1, 1.15), size=(0.12, 0.08, 0.2), yaw=0.37)
    snapshot = observe([box])
    assert len(snapshot) == 1
    np.testing.assert_allclose(snapshot[0].box.center, box.center)
    np.testing.assert_allclose(sorted(snapshot[0].box.size), sorted(box.size))
    assert math.isclose((snapshot[0].box.yaw - box.yaw) % (math.pi / 2), 0, abs_tol=1e-12)


def test_one_side_occlusion_retains_full_depth_and_actual_height():
    rear = Box((0, 1, 0.2), (0.2, 0.4, 0.4))
    front = Box((-0.05, 0, 0.01), (0.1, 0.1, 0.02))
    result = observe([rear, front], direction=(0, 1))
    rear_parts = [item.box for item in result if item.box.center[1] > 0.5]
    assert len(rear_parts) == 1
    np.testing.assert_allclose(rear_parts[0].center, (0.05, 1, 0.2), atol=1e-12)
    np.testing.assert_allclose(rear_parts[0].size, (0.1, 0.4, 0.4), atol=1e-12)


def test_disconnected_fragments_have_distinct_ids_and_preserve_depth():
    result = observe([Box((0, 1, .1), (.2, .4, .2)),
                      Box((0, 0, .01), (.04, .1, .02))], direction=(0, 1))
    assert len(result) == 3  # two physical objects yield three observations
    assert len({item.observation_id for item in result}) == 3
    rear = [item.box for item in result if item.box.center[1] > .5]
    np.testing.assert_allclose([b.center for b in rear], [(-.06, 1, .1), (.06, 1, .1)])
    np.testing.assert_allclose([b.size for b in rear], [(.08, .4, .2)] * 2)


@pytest.mark.parametrize("whole,visible,included", [
    (.06, .019, False), (.20, .03, False), (.10, .02, True),
    (.10, .02 - 1e-9, False), (.10, .02 + 1e-9, True),
    (.20, .04 - 1e-9, False), (.20, .04, True), (.20, .04 + 1e-9, True),
])
def test_each_exposed_fragment_must_pass_both_inclusive_thresholds(whole, visible, included):
    rear = Box((whole/2, 1, .1), (whole, .1, .2))
    front = Box(((whole-visible)/2, 0, .1), (whole-visible, .1, .2))
    result = observe([rear, front], direction=(0, 1))
    assert any(item.box.center[1] > .5 for item in result) == included


def test_disconnected_exposure_is_never_summed():
    result = observe([Box((0, 1, .1), (.2, .1, .2)),
                      Box((0, 0, .1), (.14, .1, .2))], direction=(0, 1))
    assert len(result) == 1
    assert result[0].box.center[1] == 0


def test_fully_hidden_and_subminimum_objects_have_no_observation():
    front = Box((0, 0, .1), (.2, .1, .2))
    assert observe([front, Box((0, 1, 2), (.1, .1, .1))], direction=(0, 1)) == observe([front], direction=(0, 1))
    assert observe([Box((0, 0, 0), (.019, .1, .1))], direction=(0, 1)) == ()


def test_filtered_occluder_still_splits_rear_object():
    result = observe([Box((0, 1, .1), (.1, .1, .2)),
                      Box((0, 0, .1), (.01, .1, .2))], direction=(0, 1))
    assert len(result) == 2
    assert all(item.box.center[1] == 1 for item in result)
    np.testing.assert_allclose([item.box.size[0] for item in result], [.045, .045])


def test_observation_is_repeatable_and_empty_scene_is_empty():
    boxes = [Box((0, 1, .1), (.2, .1, .2)), Box((0, 0, .1), (.04, .1, .2))]
    assert observe(boxes, direction=(0, 1)) == observe(boxes, direction=(0, 1))
    assert observe([]) == ()


def test_rotated_clipped_triangle_refits_yaw_instead_of_original_axes():
    # Diamond |x|+|y-1| <= .1, clipped to x >= .05. Remaining right triangle
    # has an axis-aligned minimum rectangle of area .005 (not original yaw).
    rear = Box((0, 1, .1), (math.sqrt(.02), math.sqrt(.02), .2), math.pi/4)
    front = Box((-.075, 0, .1), (.25, .1, .2))
    result = observe([rear, front], direction=(0, 1))
    remaining = [item.box for item in result if item.box.center[1] > .5]
    assert len(remaining) == 1
    assert math.isclose(remaining[0].size[0]*remaining[0].size[1], .005, abs_tol=1e-12)


def test_rotated_partial_box_uses_minimum_rectangle_not_world_aabb():
    # Half of a long thin rotated rectangle retains its long sloping edges.
    # Its minimum rectangle is aligned with those edges: retained length is
    # .15 + .04*tan(30deg), and width is .08 (analytic trapezoid fixture).
    rear = Box((0, 1, .1), (.3, .08, .2), math.pi/6)
    front = Box((-.25, 0, .1), (.5, .1, .2))
    result = observe([rear, front], direction=(0, 1))
    fitted = next(item.box for item in result if item.box.center[1] > .5)
    assert math.isclose(fitted.size[0]*fitted.size[1], .0138475208614068, abs_tol=1e-12)


def test_front_surface_order_can_change_between_projected_vertices():
    boxes = [Box((0, 0, .1), (.3, .08, .1), math.pi/4),
             Box((0, 0, .5), (.3, .08, .1), -math.pi/4)]
    result = observe(boxes, direction=(0, 1))
    assert len(result) == 2
    np.testing.assert_allclose([item.box.size[0]*item.box.size[1] for item in result], [.0152, .0152])


def test_central_strip_refits_away_from_source_yaw():
    source = Box((0, 1, .1), (math.sqrt(.02), math.sqrt(.02), .2), math.pi/4)
    left = Box((-.11, 0, .1), (.18, .1, .2))
    right = Box((.11, 0, .1), (.18, .1, .2))
    result = observe([source, left, right], direction=(0, 1))
    fitted = next(item.box for item in result if item.box.center[1] > .5)
    np.testing.assert_allclose(fitted.center, (0, 1, .1), atol=1e-12)
    np.testing.assert_allclose(fitted.size, (.04, .2, .2), atol=1e-12)
    assert abs(fitted.yaw) < 1e-12


def test_overlapping_occluder_intervals_form_a_union():
    rear = Box((0, 1, .1), (.2, .1, .2))
    front = [Box((-.025, 0, .1), (.05, .1, .2)),
             Box((0, -.2, .1), (.05, .1, .2))]
    result = observe([rear, *front], direction=(0, 1))
    remaining = [item.box for item in result if item.box.center[1] > .5]
    np.testing.assert_allclose([box.size[0] for box in remaining], [.05, .075])


def test_default_front_is_shelf_positive_x_and_height_never_unoccludes():
    rear = Box((0, 0, 2), (.1, .1, .3))
    foreground = Box((1, 0, .01), (.1, .1, .02))
    assert observe([rear, foreground])[0].box == foreground
    assert len(observe([rear, foreground])) == 1


def test_translation_and_observation_frame_rotation_preserve_geometry():
    # This fixture has only axis-aligned residuals, so its transformed answer
    # can be compared as geometry without depending on a yaw representation.
    boxes = [Box((0, 1, .1), (.2, .4, .2)), Box((0, 0, .1), (.04, .1, .2))]
    angle = .71
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s), (s, c)))
    moved = [Box((*(rotation @ b.center[:2] + (.6, -.7)), b.center[2]), b.size, angle) for b in boxes]
    original = observe(boxes, direction=(0, 1))
    rotated = observe(moved, direction=(-s, c))
    assert len(rotated) == len(original)
    for before, after in zip(original, rotated):
        np.testing.assert_allclose(after.box.center[:2], rotation @ before.box.center[:2] + (.6, -.7))
        np.testing.assert_allclose(sorted(after.box.size[:2]), sorted(before.box.size[:2]))
