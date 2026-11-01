"""Geometry checks independent of Isaac Sim startup."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shelf_object_gallery.layout import KINDS, grid_layout

BOUNDS = {
    "cylinder": ((-.04, -.04, 0), (.04, .04, .1)),
    "cube": ((-.03, -.04, 0), (.03, .04, .1)),
    "mug": ((-.047, -.035, 0), (.054, .035, .1)),
    "bottle": ((-.03, -.03, 0), (.03, .03, .135)),
    "003_cracker_box": ((-.054965, -.024059, 0), (.054965, .024059, .143037)),
    "004_sugar_box": ((-.047324, -.023047, 0), (.047324, .023047, .18)),
    "005_tomato_soup_can": ((-.042409, -.042444, 0), (.042409, .042444, .127686)),
    "002_master_chef_can": ((-.042446, -.042407, 0), (.042446, .042407, .116206)),
}


@pytest.mark.parametrize("rows,cols,gap", [(1, 4, .05), (2, 4, .05), (3, 6, .01)])
def test_full_bounds_fit_and_grid_centres_are_regular(rows, cols, gap):
    placements = grid_layout(rows, cols, BOUNDS, gap=gap)
    boxes = []
    centers = []
    for item in placements:
        low, high = BOUNDS[item.kind]
        lo = tuple(a + p for a, p in zip(low, item.position))
        hi = tuple(b + p for b, p in zip(high, item.position))
        assert -.18 - 1e-9 <= lo[0] < hi[0] <= .18 + 1e-9
        assert -.48 - 1e-9 <= lo[1] < hi[1] <= .48 + 1e-9
        assert lo[2] == pytest.approx(1.05)
        boxes.append((lo, hi))
        centers.append(tuple((a + b) / 2 for a, b in zip(lo, hi)))
    for i, (lo, hi) in enumerate(boxes):
        for lo2, hi2 in boxes[i + 1:]:
            assert any(hi[axis] + gap <= lo2[axis] + 1e-9 or hi2[axis] + gap <= lo[axis] + 1e-9
                       for axis in (0, 1))
    assert sum(p[0] for p in centers) == pytest.approx(0, abs=1e-12)
    assert sum(p[1] for p in centers) == pytest.approx(0, abs=1e-12)
    if rows > 1:
        assert centers[0][0] > centers[cols][0]
    if cols > 1:
        assert centers[0][1] < centers[1][1]


def test_reject_overcrowding_instead_of_scaling_objects():
    with pytest.raises(ValueError, match="object scale remains 1.0"):
        grid_layout(4, 4, BOUNDS)


def test_two_by_four_contains_all_eight_kinds_once_including_ycb():
    placements = grid_layout(2, 4, BOUNDS)
    assert tuple(p.kind for p in placements) == KINDS
    assert len(set(p.kind for p in placements)) == 8
    assert tuple(p.kind for p in grid_layout(2, 8, BOUNDS, gap=0)) == KINDS * 2


@pytest.mark.parametrize("rows,cols,gap", [(0, 4, .05), (1, -1, .05), (1, 4, -.01), (1, 4, float('nan'))])
def test_invalid_grid(rows, cols, gap):
    with pytest.raises(ValueError):
        grid_layout(rows, cols, BOUNDS, gap=gap)
