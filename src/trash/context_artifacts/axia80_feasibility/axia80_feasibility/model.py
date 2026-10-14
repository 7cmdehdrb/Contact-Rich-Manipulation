"""Generate a local UR5e--Axia80--tool URDF without a Nucleus dependency.

Only files below the caller-provided generated directory are written.  The
Universal Robots description in ``src/Universal_Robots_ROS2_Description`` is
treated as read-only source material.
"""

from __future__ import annotations

import json
import math
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SENSOR_LINK_NAME = "axia80_link"
TOOL_LINK_NAME = "payload_tool_link"

# User-specified Axia80 envelope.
SENSOR_RADIUS_M = 0.041
SENSOR_HEIGHT_M = 0.0254

# The sensor is an idealized measurement body.  Its own mass is excluded by
# reading the downstream tool joint, but a small non-zero mass keeps PhysX's
# rigid-body representation well conditioned.
SENSOR_MASS_KG = 0.001

# Broad XY face, thin Z dimension: a simple spatula-like placeholder for the
# eventual Inspire Robot tool.
TOOL_SIZE_M = (0.24, 0.16, 0.012)
TOOL_MASS_KG = 0.20
TOOL_GAP_M = 0.001
PAYLOAD_SIZE_M = 0.040

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# This pose points the tool/sensor local +Z axis along world +Z.  Consequently
# the tool's broad local XY face is parallel to the ground and its +normal is up.
ARM_POSE = dict(
    zip(
        ARM_JOINT_NAMES,
        (0.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0, 0.0),
    )
)


def repo_root() -> Path:
    """Locate this checkout from the installed or source package path."""
    for path in Path(__file__).resolve().parents:
        if (path / "IsaacLab").is_dir() and (path / "src/Universal_Robots_ROS2_Description").is_dir():
            return path
    raise FileNotFoundError("Cannot locate the grad checkout; pass --repo_root explicitly.")


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


def _set_origin(element: ET.Element, transform: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    origin.set("xyz", " ".join(f"{value:.17g}" for value in transform[:3, 3]))
    origin.set("rpy", " ".join(f"{value:.17g}" for value in rpy))


def _collapse_empty_frames(robot: ET.Element) -> None:
    """Remove massless TF-only UR links while preserving child transforms."""
    for link in list(robot.findall("link")):
        name = link.get("name")
        if name == "base_link" or len(link):
            continue
        incoming = next(
            joint for joint in robot.findall("joint") if joint.find("child").get("link") == name
        )
        if incoming.get("type") != "fixed":
            raise ValueError(f"Cannot remove moving empty frame: {name}")
        parent = incoming.find("parent").get("link")
        transform = _origin_matrix(incoming.find("origin"))
        for joint in robot.findall("joint"):
            if joint.find("parent").get("link") == name:
                joint.find("parent").set("link", parent)
                _set_origin(joint, transform @ _origin_matrix(joint.find("origin")))
        robot.remove(incoming)
        robot.remove(link)

    # UR descriptions put the physical base geometry below a fixed
    # ``base_link_inertia`` child.  Fold it into the fixed root to avoid an
    # unnecessary rigid body without changing any downstream transform.
    root = robot.find("link[@name='base_link']")
    physical = robot.find("link[@name='base_link_inertia']")
    if root is not None and physical is not None and len(root) == 0:
        incoming = next(
            joint
            for joint in robot.findall("joint")
            if joint.find("child").get("link") == "base_link_inertia"
        )
        transform = _origin_matrix(incoming.find("origin"))
        for source in physical:
            element = ET.fromstring(ET.tostring(source))
            _set_origin(element, transform @ _origin_matrix(element.find("origin")))
            root.append(element)
        for joint in robot.findall("joint"):
            if joint.find("parent").get("link") == "base_link_inertia":
                joint.find("parent").set("link", "base_link")
                _set_origin(joint, transform @ _origin_matrix(joint.find("origin")))
        robot.remove(incoming)
        robot.remove(physical)


def _expand_ur5e(description: Path) -> ET.Element:
    """Expand the local UR xacro without ROS package discovery."""
    import xacro

    with tempfile.TemporaryDirectory(prefix="ur5e-axia80-xacro-") as scratch_dir:
        scratch = Path(scratch_dir)
        for source in (description / "urdf").rglob("*.xacro"):
            destination = scratch / source.relative_to(description)
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_text().replace(
                "$(find ur_description)/urdf", str(scratch / "urdf")
            ).replace("$(find ur_description)", str(description))
            destination.write_text(content)

        wrapper = ET.Element(
            "robot", {"name": "ur5e_axia80_tool", "xmlns:xacro": "http://www.ros.org/wiki/xacro"}
        )
        ET.SubElement(
            wrapper,
            "xacro:include",
            {"filename": str(scratch / "urdf/ur_macro.xacro")},
        )
        ET.SubElement(wrapper, "link", {"name": "world"})
        macro = ET.SubElement(
            wrapper,
            "xacro:ur_robot",
            {
                "name": "ur5e",
                "tf_prefix": "",
                "parent": "world",
                "ur_type": "ur5e",
                "generate_ros2_control_tag": "false",
                "joint_limits_parameters_file": str(description / "config/ur5e/joint_limits.yaml"),
                "kinematics_parameters_file": str(description / "config/ur5e/default_kinematics.yaml"),
                "physical_parameters_file": str(description / "config/ur5e/physical_parameters.yaml"),
                "visual_parameters_file": str(description / "config/ur5e/visual_parameters.yaml"),
            },
        )
        ET.SubElement(macro, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        entry = scratch / "robot.xacro"
        ET.ElementTree(wrapper).write(entry, encoding="unicode")
        robot = ET.fromstring(xacro.process_file(str(entry)).toxml())

    for element in list(robot):
        if element.tag == "link" and element.get("name") == "world":
            robot.remove(element)
        elif element.tag == "joint" and element.find("parent").get("link") == "world":
            robot.remove(element)
        elif element.tag not in ("link", "joint", "material"):
            robot.remove(element)
    for mesh in robot.iter("mesh"):
        mesh.set(
            "filename",
            mesh.get("filename").replace("package://ur_description/", str(description) + "/"),
        )
    return robot


def _add_inertial(
    link: ET.Element,
    *,
    mass: float,
    origin_xyz: tuple[float, float, float],
    inertia: tuple[float, float, float],
) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": " ".join(map(str, origin_xyz)), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ixx, iyy, izz = inertia
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(ixx),
            "ixy": "0",
            "ixz": "0",
            "iyy": str(iyy),
            "iyz": "0",
            "izz": str(izz),
        },
    )


def _add_geometry(
    link: ET.Element,
    *,
    shape: str,
    attributes: dict[str, str],
    origin_xyz: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> None:
    for role in ("visual", "collision"):
        element = ET.SubElement(link, role)
        ET.SubElement(element, "origin", {"xyz": " ".join(map(str, origin_xyz)), "rpy": "0 0 0"})
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, shape, attributes)
        if role == "visual":
            material = ET.SubElement(element, "material", {"name": f"{link.get('name')}_material"})
            ET.SubElement(material, "color", {"rgba": " ".join(map(str, color))})


def _append_axia80_and_tool(robot: ET.Element) -> None:
    if robot.find("link[@name='tool0']") is None:
        raise ValueError("Expanded UR5e is missing its tool0 frame")

    sensor = ET.SubElement(robot, "link", {"name": SENSOR_LINK_NAME})
    radial_inertia = SENSOR_MASS_KG * (3.0 * SENSOR_RADIUS_M**2 + SENSOR_HEIGHT_M**2) / 12.0
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
    mount = ET.SubElement(robot, "joint", {"name": "ur5e_to_axia80_joint", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "tool0"})
    ET.SubElement(mount, "child", {"link": SENSOR_LINK_NAME})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": f"0 0 {0.5 * SENSOR_HEIGHT_M}", "rpy": "0 0 0"},
    )

    # The tool link frame deliberately coincides with the sensor center.  Its
    # geometry and COM are offset downstream.  Thus the parent sensor frame and
    # child tool frame are identical, making the wrench axes/origin unambiguous
    # even though Isaac Lab and the lower PhysX API describe the returned joint
    # wrench frame differently.
    tool_center_z = 0.5 * SENSOR_HEIGHT_M + TOOL_GAP_M + 0.5 * TOOL_SIZE_M[2]
    tool = ET.SubElement(robot, "link", {"name": TOOL_LINK_NAME})
    size_x, size_y, size_z = TOOL_SIZE_M
    _add_inertial(
        tool,
        mass=TOOL_MASS_KG,
        origin_xyz=(0.0, 0.0, tool_center_z),
        inertia=(
            TOOL_MASS_KG * (size_y**2 + size_z**2) / 12.0,
            TOOL_MASS_KG * (size_x**2 + size_z**2) / 12.0,
            TOOL_MASS_KG * (size_x**2 + size_y**2) / 12.0,
        ),
    )
    _add_geometry(
        tool,
        shape="box",
        attributes={"size": " ".join(map(str, TOOL_SIZE_M))},
        origin_xyz=(0.0, 0.0, tool_center_z),
        color=(0.12, 0.38, 0.88, 1.0),
    )
    measurement_joint = ET.SubElement(
        robot, "joint", {"name": "axia80_measurement_joint", "type": "fixed"}
    )
    ET.SubElement(measurement_joint, "parent", {"link": SENSOR_LINK_NAME})
    ET.SubElement(measurement_joint, "child", {"link": TOOL_LINK_NAME})
    ET.SubElement(measurement_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})


def build_model(root: Path, output: Path) -> tuple[Path, dict[str, object]]:
    """Generate the combined URDF and return its path plus experiment metadata."""
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    description = root / "src/Universal_Robots_ROS2_Description"
    if not description.is_dir():
        raise FileNotFoundError(f"UR description directory does not exist: {description}")

    robot = _expand_ur5e(description)
    _append_axia80_and_tool(robot)
    _collapse_empty_frames(robot)

    links = {link.get("name") for link in robot.findall("link")}
    if SENSOR_LINK_NAME not in links or TOOL_LINK_NAME not in links:
        raise RuntimeError("Axia80 or tool link was removed during URDF simplification")
    for mesh in robot.iter("mesh"):
        if not Path(mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))

    metadata: dict[str, object] = {
        "sensor_link_name": SENSOR_LINK_NAME,
        "tool_link_name": TOOL_LINK_NAME,
        "measurement_joint_name": "axia80_measurement_joint",
        "sensor_radius_m": SENSOR_RADIUS_M,
        "sensor_height_m": SENSOR_HEIGHT_M,
        "sensor_mass_kg": SENSOR_MASS_KG,
        "tool_size_m": TOOL_SIZE_M,
        "tool_mass_kg": TOOL_MASS_KG,
        "tool_gap_m": TOOL_GAP_M,
        "tool_center_z_in_sensor_m": 0.5 * SENSOR_HEIGHT_M + TOOL_GAP_M + 0.5 * TOOL_SIZE_M[2],
        "tool_top_z_in_sensor_m": 0.5 * SENSOR_HEIGHT_M + TOOL_GAP_M + TOOL_SIZE_M[2],
        "payload_size_m": PAYLOAD_SIZE_M,
        "arm_pose": ARM_POSE,
        "wrench_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "wrench_frame": "axia80 sensor center; local +Z is the upward tool-face normal",
    }
    ET.indent(robot, space="  ")
    urdf_path = output / "ur5e_axia80_tool.urdf"
    content = ET.tostring(robot, encoding="unicode")
    if not urdf_path.exists() or urdf_path.read_text() != content:
        urdf_path.write_text(content)
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return urdf_path, metadata

