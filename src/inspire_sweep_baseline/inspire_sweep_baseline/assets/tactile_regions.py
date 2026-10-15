"""Read-only tactile footprints derived from the unchanged robot collision USD."""

import json
from pathlib import Path

from .robot import PALM_SENSOR_BODY_NAMES


_PATH = Path(__file__).resolve().parent / "data" / "tactile_regions.json"
_METADATA = json.loads(_PATH.read_text())
TACTILE_REGIONS = tuple(_METADATA["regions"])
if tuple(region["name"] for region in TACTILE_REGIONS) != PALM_SENSOR_BODY_NAMES:
    raise RuntimeError("Tactile geometry metadata must match the policy sensor order")
TACTILE_PARENT_BODY_NAMES = tuple(dict.fromkeys(region["parent_name"] for region in TACTILE_REGIONS))


def parent_sensor_name(parent):
    return "tactile_region_" + parent
