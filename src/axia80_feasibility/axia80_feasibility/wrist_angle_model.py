"""Model metadata for the fixed-mass wrist roll/tilt angle sweep."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from .cantilever_model import CANTILEVER_ARM_POSE, build_cantilever_model, repo_root
from .tilted_cantilever_model import (
    TILTED_DYNAMIC_FRICTION,
    TILTED_STATIC_FRICTION,
)

DEFAULT_ROLL_ANGLES_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
DEFAULT_TILT_ANGLES_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
DEFAULT_PAYLOAD_MASS_G = 500
ROLL_JOINT_NAME = "wrist_3_joint"
TILT_JOINT_NAME = "wrist_2_joint"


def _validated_angles(values: Sequence[float], name: str) -> tuple[float, ...]:
    angles = tuple(float(value) for value in values)
    if not angles or not all(math.isfinite(value) for value in angles):
        raise ValueError(f"{name} must contain finite values")
    if tuple(sorted(set(angles))) != angles:
        raise ValueError(f"{name} must be strictly increasing and unique")
    if any(abs(value) > 30.0 for value in angles):
        raise ValueError(f"{name} is limited to +/-30 degrees for this contact experiment")
    return angles


def angle_grid(
    roll_angles_deg: Sequence[float],
    tilt_angles_deg: Sequence[float],
) -> list[tuple[float, float]]:
    """Return ``(roll, tilt)`` pairs with tilt rows and roll columns."""
    rolls = _validated_angles(roll_angles_deg, "roll_angles_deg")
    tilts = _validated_angles(tilt_angles_deg, "tilt_angles_deg")
    return [(roll, tilt) for tilt in tilts for roll in rolls]


def arm_pose_for_angles(roll_deg: float, tilt_deg: float) -> dict[str, float]:
    """Apply roll/tilt offsets to the established horizontal cantilever pose."""
    if not math.isfinite(roll_deg) or not math.isfinite(tilt_deg):
        raise ValueError("roll_deg and tilt_deg must be finite")
    pose = dict(CANTILEVER_ARM_POSE)
    pose[ROLL_JOINT_NAME] += math.radians(roll_deg)
    pose[TILT_JOINT_NAME] += math.radians(tilt_deg)
    return pose


def build_wrist_angle_model(
    root: Path,
    output: Path,
) -> tuple[Path, dict[str, object]]:
    """Build a separate narrow cantilever model for the angle-grid experiment."""
    urdf_path, base_model = build_cantilever_model(root, output)
    model = dict(base_model)
    model.update(
        {
            "configuration": "fixed_mass_wrist_roll_tilt_angle_sweep",
            "base_arm_pose": dict(CANTILEVER_ARM_POSE),
            "arm_pose": dict(CANTILEVER_ARM_POSE),
            "roll_joint_name": ROLL_JOINT_NAME,
            "tilt_joint_name": TILT_JOINT_NAME,
            "roll_definition": "wrist_3 offset about the sensor/tool +Z cantilever axis",
            "tilt_definition": "wrist_2 offset from the horizontal cantilever pose",
            "default_roll_angles_deg": list(DEFAULT_ROLL_ANGLES_DEG),
            "default_tilt_angles_deg": list(DEFAULT_TILT_ANGLES_DEG),
            "default_payload_mass_g": DEFAULT_PAYLOAD_MASS_G,
            "static_friction": TILTED_STATIC_FRICTION,
            "dynamic_friction": TILTED_DYNAMIC_FRICTION,
            "nominal_payload_force_sensor_frame": {
                "Fx": "m*g*cos(roll)*cos(tilt)",
                "Fy": "-m*g*sin(roll)*cos(tilt)",
                "Fz": "-m*g*sin(tilt)",
            },
            "angle_grid_order": "tilt-major rows, roll-minor columns",
        }
    )
    serialized = json.dumps(model, indent=2) + "\n"
    (output / "cantilever_model.json").write_text(serialized)
    (output / "wrist_angle_sweep_model.json").write_text(serialized)
    return urdf_path, model


__all__ = [
    "DEFAULT_PAYLOAD_MASS_G",
    "DEFAULT_ROLL_ANGLES_DEG",
    "DEFAULT_TILT_ANGLES_DEG",
    "ROLL_JOINT_NAME",
    "TILT_JOINT_NAME",
    "angle_grid",
    "arm_pose_for_angles",
    "build_wrist_angle_model",
    "repo_root",
]
