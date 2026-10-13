"""External shelf location; robot USD and target geometry belong to this package."""

import os
import re
from pathlib import Path


def _resolve(value):
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        return value.rstrip("/")
    return str(Path(value).expanduser().resolve())


def shelf_usd_path():
    # Existing asset-location environment variables remain usable on lab PCs.
    # This reads configuration strings only, without importing Sweep-Policy.
    override = os.environ.get("INSPIRE_SWEEP_SHELF_USD_PATH") or os.environ.get("SWEEP_POLICY_SHELF_USD_PATH")
    if override:
        return _resolve(override)
    root = os.environ.get("INSPIRE_SWEEP_ASSET_ROOT") or os.environ.get("SWEEP_POLICY_ASSET_ROOT")
    root = _resolve(root or "omniverse://192.168.0.13/Library/Shelf")
    return f"{root}/Arena/Collected_speedrack_shape/speedrack_shape.usd"


SHELF_USD_PATH = shelf_usd_path()
