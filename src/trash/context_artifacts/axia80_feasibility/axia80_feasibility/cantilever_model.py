"""Build the additional thin-face-mounted Axia80 cantilever experiment.

The original broad-face-mounted model in :mod:`axia80_feasibility.model` is
left unchanged.  This module generates a separate URDF in which the thin end
face of the rectangular tool is attached to the sensor and the tool projects
horizontally like a gripper finger.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .model import (
    ARM_JOINT_NAMES,
    PAYLOAD_SIZE_M,
    SENSOR_HEIGHT_M,
    SENSOR_LINK_NAME,
    SENSOR_MASS_KG,
    SENSOR_RADIUS_M,
    TOOL_LINK_NAME,
    _add_geometry,
    _add_inertial,
    _collapse_empty_frames,
    _expand_ur5e,
    repo_root,
)

# Dimensions are expressed in the common sensor/tool measurement frame.
# X: tool thickness / broad-face normal, Y: width, Z: cantilever length.
# The width is kept inside a narrower envelope than the original 0.160 m
# prototype to avoid visual/self-overlap around the UR wrist while retaining
# 40 mm of lateral margin for the centered 40 mm payload.
CANTILEVER_TOOL_SIZE_M = (0.012, 0.12, 0.24)
CANTILEVER_TOOL_MASS_KG = 0.20
CANTILEVER_TOOL_GAP_M = 0.001

# Relative to the original pose, wrist_2 is rotated by +90 degrees.  This
# maps sensor +X to world +Z and sensor +Z to world -Y.
CANTILEVER_ARM_POSE = dict(
    zip(
        ARM_JOINT_NAMES,
        (0.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, math.pi, 0.0),
    )
)


def _append_axia80_and_cantilever_tool(robot: ET.Element) -> None:
    """Append the sensor and edge-mounted tool without changing the base model."""
    if robot.find("link[@name='tool0']") is None:
        raise ValueError("Expanded UR5e is missing its tool0 frame")

    sensor = ET.SubElement(robot, "link", {"name": SENSOR_LINK_NAME})
    radial_inertia = SENSOR_MASS_KG * (
        3.0 * SENSOR_RADIUS_M**2 + SENSOR_HEIGHT_M**2
    ) / 12.0
    axial_inertia = 0.5 * SENSOR_MASS_KG * SENSOR_RADIUS_M**2
    _add_inertial(
        sensor,
        mass=SENSOR_MASS_KG,
        origin_xyz=(0.0, 0.0, 0.0),
        inertia=(radial_inertia, radial_inertia, axial_inertia),
    )
    _add_geometry(
        sensor,
        shape="cylinder",
        attributes={"radius": str(SENSOR_RADIUS_M), "length": str(SENSOR_HEIGHT_M)},
        origin_xyz=(0.0, 0.0, 0.0),
        color=(0.90, 0.32, 0.08, 1.0),
    )
    mount = ET.SubElement(
        robot,
        "joint",
        {"name": "ur5e_to_axia80_cantilever_joint", "type": "fixed"},
    )
    ET.SubElement(mount, "parent", {"link": "tool0"})
    ET.SubElement(mount, "child", {"link": SENSOR_LINK_NAME})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": f"0 0 {0.5 * SENSOR_HEIGHT_M}", "rpy": "0 0 0"},
    )

    # The narrow 0.012 x 0.12 m end face is normal to sensor +Z and starts
    # just beyond the cylindrical sensor face.  The broad YZ face therefore
    # has normal +X, which CANTILEVER_ARM_POSE maps to world up.
    size_x, size_y, size_z = CANTILEVER_TOOL_SIZE_M
    tool_center_z = 0.5 * SENSOR_HEIGHT_M + CANTILEVER_TOOL_GAP_M + 0.5 * size_z
    tool = ET.SubElement(robot, "link", {"name": TOOL_LINK_NAME})
    _add_inertial(
        tool,
        mass=CANTILEVER_TOOL_MASS_KG,
        origin_xyz=(0.0, 0.0, tool_center_z),
        inertia=(
            CANTILEVER_TOOL_MASS_KG * (size_y**2 + size_z**2) / 12.0,
            CANTILEVER_TOOL_MASS_KG * (size_x**2 + size_z**2) / 12.0,
            CANTILEVER_TOOL_MASS_KG * (size_x**2 + size_y**2) / 12.0,
        ),
    )
    _add_geometry(
        tool,
        shape="box",
        attributes={"size": " ".join(map(str, CANTILEVER_TOOL_SIZE_M))},
        origin_xyz=(0.0, 0.0, tool_center_z),
        color=(0.10, 0.42, 0.88, 1.0),
    )
    measurement_joint = ET.SubElement(
        robot,
        "joint",
        {"name": "axia80_cantilever_measurement_joint", "type": "fixed"},
    )
    ET.SubElement(measurement_joint, "parent", {"link": SENSOR_LINK_NAME})
    ET.SubElement(measurement_joint, "child", {"link": TOOL_LINK_NAME})
    ET.SubElement(measurement_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def build_cantilever_model(root: Path, output: Path) -> tuple[Path, dict[str, object]]:
    """Generate the independent thin-face-mounted URDF and its metadata."""
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    description = root / "src/Universal_Robots_ROS2_Description"
    if not description.is_dir():
        raise FileNotFoundError(f"UR description directory does not exist: {description}")

    robot = _expand_ur5e(description)
    robot.set("name", "ur5e_axia80_cantilever_tool")
    _append_axia80_and_cantilever_tool(robot)
    _collapse_empty_frames(robot)

    links = {link.get("name") for link in robot.findall("link")}
    if SENSOR_LINK_NAME not in links or TOOL_LINK_NAME not in links:
        raise RuntimeError("Axia80 or cantilever tool link disappeared during URDF simplification")
    for mesh in robot.iter("mesh"):
        if not Path(mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))

    size_x, size_y, size_z = CANTILEVER_TOOL_SIZE_M
    tool_start_z = 0.5 * SENSOR_HEIGHT_M + CANTILEVER_TOOL_GAP_M
    tool_center_z = tool_start_z + 0.5 * size_z
    payload_center_x = 0.5 * size_x + 0.5 * PAYLOAD_SIZE_M
    metadata: dict[str, object] = {
        "configuration": "thin_face_cantilever",
        "sensor_link_name": SENSOR_LINK_NAME,
        "tool_link_name": TOOL_LINK_NAME,
        "measurement_joint_name": "axia80_cantilever_measurement_joint",
        "sensor_radius_m": SENSOR_RADIUS_M,
        "sensor_height_m": SENSOR_HEIGHT_M,
        "sensor_mass_kg": SENSOR_MASS_KG,
        "sensor_axis_in_measurement_frame": [0.0, 0.0, 1.0],
        "tool_size_sensor_frame_m": CANTILEVER_TOOL_SIZE_M,
        "tool_mass_kg": CANTILEVER_TOOL_MASS_KG,
        "tool_gap_m": CANTILEVER_TOOL_GAP_M,
        "tool_mount_face_size_m": [size_x, size_y],
        "tool_mount_face_outward_normal_axis": "-Z",
        "tool_extension_axis": "+Z",
        "tool_broad_face_size_m": [size_y, size_z],
        "tool_broad_face_normal_axis": "+X",
        "tool_start_z_in_sensor_m": tool_start_z,
        "tool_center_z_in_sensor_m": tool_center_z,
        "payload_size_m": PAYLOAD_SIZE_M,
        "payload_center_in_sensor_m": [payload_center_x, 0.0, tool_center_z],
        "payload_bending_lever_arm_m": tool_center_z,
        "arm_pose": CANTILEVER_ARM_POSE,
        "wrench_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "wrench_frame": "Axia80 center; +X is upward broad-face normal; +Z is cantilever axis",
        "expected_payload_channels": {
            "normal_force": "+Fx = mass * gravity",
            "bending_moment": "+My = mass * gravity * payload_bending_lever_arm_m",
        },
    }
    ET.indent(robot, space="  ")
    urdf_path = output / "ur5e_axia80_cantilever_tool.urdf"
    content = ET.tostring(robot, encoding="unicode")
    if not urdf_path.exists() or urdf_path.read_text() != content:
        urdf_path.write_text(content)
    (output / "cantilever_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return urdf_path, metadata


__all__ = [
    "CANTILEVER_ARM_POSE",
    "CANTILEVER_TOOL_GAP_M",
    "CANTILEVER_TOOL_MASS_KG",
    "CANTILEVER_TOOL_SIZE_M",
    "build_cantilever_model",
    "repo_root",
]
