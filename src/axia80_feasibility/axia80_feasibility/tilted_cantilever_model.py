"""Configuration for the additional 10-degree wrist-tilted cantilever test."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .cantilever_model import CANTILEVER_ARM_POSE, build_cantilever_model, repo_root

WRIST3_TILT_DEG = 10.0
WRIST3_TILT_RAD = math.radians(WRIST3_TILT_DEG)
TILTED_STATIC_FRICTION = 2.0
TILTED_DYNAMIC_FRICTION = 1.5

TILTED_ARM_POSE = dict(CANTILEVER_ARM_POSE)
TILTED_ARM_POSE["wrist_3_joint"] = WRIST3_TILT_RAD


def build_tilted_cantilever_model(
    root: Path,
    output: Path,
) -> tuple[Path, dict[str, object]]:
    """Build the narrow tool and attach tilt-specific analytical metadata."""
    urdf_path, base_model = build_cantilever_model(root, output)
    model = dict(base_model)
    lever_arm = float(model["payload_bending_lever_arm_m"])
    payload_normal_offset = float(model["payload_center_in_sensor_m"][0])
    cosine = math.cos(WRIST3_TILT_RAD)
    sine = math.sin(WRIST3_TILT_RAD)
    model.update(
        {
            "configuration": "thin_face_cantilever_wrist3_tilt_10deg",
            "arm_pose": TILTED_ARM_POSE,
            "wrist3_tilt_deg": WRIST3_TILT_DEG,
            "wrist3_tilt_rad": WRIST3_TILT_RAD,
            "static_friction": TILTED_STATIC_FRICTION,
            "dynamic_friction": TILTED_DYNAMIC_FRICTION,
            "minimum_static_friction_for_no_slide": math.tan(WRIST3_TILT_RAD),
            "expected_world_up_in_sensor_frame": [cosine, -sine, 0.0],
            "expected_broad_normal_world_z": cosine,
            "expected_sensor_axis_world_z": 0.0,
            "expected_payload_slopes_per_kg": {
                "Fx_N_per_kg": 9.81 * cosine,
                "Fy_N_per_kg": -9.81 * sine,
                "Fz_N_per_kg": 0.0,
                "Mx_Nm_per_kg": 9.81 * lever_arm * sine,
                "My_Nm_per_kg": 9.81 * lever_arm * cosine,
                "Mz_Nm_per_kg": -9.81 * payload_normal_offset * sine,
            },
            "expected_distribution": (
                "gravity is resolved across +Fx/-Fy and the moment across "
                "+Mx/+My/-Mz in the Axia80 frame"
            ),
        }
    )
    serialized = json.dumps(model, indent=2) + "\n"
    # Replace the generic metadata in this tilt-only generated directory so
    # every nearby metadata file describes the same initial joint pose.
    (output / "cantilever_model.json").write_text(serialized)
    (output / "cantilever_tilt10_model.json").write_text(serialized)
    return urdf_path, model


__all__ = [
    "TILTED_ARM_POSE",
    "TILTED_DYNAMIC_FRICTION",
    "TILTED_STATIC_FRICTION",
    "WRIST3_TILT_DEG",
    "WRIST3_TILT_RAD",
    "build_tilted_cantilever_model",
    "repo_root",
]
