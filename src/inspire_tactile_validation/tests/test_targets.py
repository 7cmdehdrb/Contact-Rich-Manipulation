"""Probe planning checks that need neither Isaac Sim nor Isaac Lab."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "inspire_tactile_validation/targets.py"
_SPEC = importlib.util.spec_from_file_location("_tactile_validation_targets", _PATH)
targets_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(targets_module)


def test_targets_cover_all_regions_in_production_order():
    regions = targets_module.load_regions()
    targets = targets_module.build_targets()
    assert len(targets) == 85
    assert Counter(target["channel"] for target in targets) == {channel: 5 for channel in range(17)}
    assert [target["name"] for target in targets[::5]] == [region["name"] for region in regions]
    assert all(targets[channel * 5]["location"] == "center" for channel in range(17))
    for target in targets:
        region = regions[target["channel"]]
        assert target["parent"] == region["parent_name"]
        assert target["normal"] == [0.0, 0.0, 1.0]
        assert target["local_point"][2] == region["pad_bounds_m"]["max"][2]
    # The manifest can be emitted without a custom encoder.
    assert json.loads(json.dumps(targets)) == targets


@pytest.mark.parametrize("inset", [0.01, 0.2, 0.7])
def test_every_target_is_inside_actual_convex_pad_polygon(inset):
    regions = targets_module.load_regions()
    targets = targets_module.build_targets(inset)
    for target in targets:
        x, y, _ = target["local_point"]
        planes = regions[target["channel"]]["footprint_halfspaces"]
        assert all(a * x + b * y + c < 0.0 for a, b, c in planes), target
    for channel in range(17):
        points = [tuple(target["local_point"]) for target in targets[channel * 5:channel * 5 + 5]]
        assert len(set(points)) == 5


def test_corner_rays_stop_at_clipped_polygon_before_bbox_corner():
    # A diamond's bounding-box corners lie outside its footprint. Four direct
    # bbox samples would miss; ray intersection finds each sloping edge first.
    region = {
        "name": "diamond", "parent_name": "parent", "normal_axis": [0, 0, 1],
        "pad_bounds_m": {"min": [-1, -1, -0.1], "max": [1, 1, 0.1]},
        "footprint_halfspaces": [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
    }
    targets = targets_module.build_targets(0.2, regions=[region])
    for target in targets[1:]:
        x, y, _ = target["local_point"]
        assert abs(x) == pytest.approx(0.4)
        assert abs(y) == pytest.approx(0.4)


def test_larger_inset_moves_corners_towards_same_center():
    outer = targets_module.build_targets(0.2)
    inner = targets_module.build_targets(0.6)
    for channel in range(17):
        center = outer[channel * 5]["local_point"]
        for offset in range(1, 5):
            first = outer[channel * 5 + offset]["local_point"]
            second = inner[channel * 5 + offset]["local_point"]
            for axis in range(2):
                assert second[axis] - center[axis] == pytest.approx((first[axis] - center[axis]) * 0.5)


@pytest.mark.parametrize("inset", [-1, 0, 1, 1.1, math.nan, math.inf])
def test_invalid_corner_insets_are_rejected(inset):
    with pytest.raises(ValueError, match="corner_inset"):
        targets_module.build_targets(inset)


def test_source_metadata_is_read_without_modification():
    path = targets_module.DEFAULT_REGION_PATH
    before = path.read_bytes()
    targets_module.build_targets()
    assert path.read_bytes() == before
