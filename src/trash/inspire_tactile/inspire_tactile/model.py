"""Asset preparation and kinematics, usable without launching Isaac Sim.

Only generated files are written. The supplied URDF and UR description submodule
remain the source of geometry, inertia, joint axes, limits and mimic ratios.
"""

from __future__ import annotations

import copy
import json
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Fixed IK solution for the coaxial mount: preserve the original hand's world
# pose (x=.4919, y=.1333, z=.4629), keeping the pedestal clear of the arm.
ARM_POSE = dict(zip(
    ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
    (0.0, -1.08055181288727, 1.01636329476067, 0.06418851792149663,
     -np.pi / 2, np.pi / 2),
))
SENSOR_NAMES = (
    "palm_force_sensor",
    *(f"thumb_force_sensor_{i}" for i in range(1, 5)),
    *(f"{finger}_force_sensor_{i}" for finger in ("index", "middle", "ring", "little")
      for i in range(1, 4)),
)
HAND_PREFIX = "inspire_"
OPEN = {"left_thumb_1_joint": 0.2, "left_thumb_2_joint": 0.0,
        **{f"left_{f}_1_joint": 0.0 for f in ("index", "middle", "ring", "little")}}
# Fully flexed motor targets within the source URDF limits. Finite-torque
# drives stop against the object; actual joints are never teleported closed.
CLOSED = {"left_thumb_1_joint": 1.1641, "left_thumb_2_joint": 0.5864,
          **{f"left_{f}_1_joint": 1.4381 for f in ("index", "middle", "ring", "little")}}


def repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "IsaacLab").is_dir() and (path / "src/inspire_robot").is_dir():
            return path
    raise FileNotFoundError("Cannot locate grad checkout; pass --repo_root explicitly.")


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


def forward_kinematics(robot: ET.Element, positions: dict[str, float]) -> dict[str, np.ndarray]:
    joints = robot.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    roots = {link.get("name") for link in robot.findall("link")} - children
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root, got {roots}")
    poses = {roots.pop(): np.eye(4)}
    pending = joints.copy()
    while pending:
        old_count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            local = origin_matrix(joint.find("origin"))
            if joint.get("type") in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    axis * positions.get(joint.get("name"), 0.0)
                ).as_matrix()
                local = local @ motion
            poses[joint.find("child").get("link")] = poses[parent] @ local
            pending.remove(joint)
        if len(pending) == old_count:
            raise ValueError("Disconnected or cyclic URDF joints")
    return poses


def hand_targets(hand: ET.Element, master_positions: dict[str, float]) -> dict[str, float]:
    """Expand the six motor targets using the actual URDF mimic graph."""
    positions = dict(master_positions)
    movable = [j for j in hand.findall("joint") if j.get("type") != "fixed"]
    pending = [j for j in movable if j.get("name") not in positions]
    while pending:
        old_count = len(pending)
        for joint in pending[:]:
            mimic = joint.find("mimic")
            if mimic is None or mimic.get("joint") not in positions:
                continue
            positions[joint.get("name")] = (
                positions[mimic.get("joint")] * float(mimic.get("multiplier", "1"))
                + float(mimic.get("offset", "0"))
            )
            pending.remove(joint)
        if old_count == len(pending):
            raise ValueError("Unresolved hand joints or cyclic mimic graph")
    for joint in movable:
        limit = joint.find("limit")
        value = positions[joint.get("name")]
        if not float(limit.get("lower")) <= value <= float(limit.get("upper")):
            raise ValueError(f"Target {value} outside limits for {joint.get('name')}")
    return {HAND_PREFIX + name: value for name, value in positions.items()}


def _set_origin(element: ET.Element, transform: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    origin.set("xyz", " ".join(map(str, transform[:3, 3])))
    origin.set("rpy", " ".join(map(str, rpy)))


def collapse_empty_arm_frames(robot: ET.Element) -> None:
    """Remove massless TF-only UR frames without merging any sensor bodies.

    Keeping empty frames as rigid bodies makes the importer invent mass/inertia
    and broken visual references. Compose their transforms into child joints.
    """
    for link in list(robot.findall("link")):
        name = link.get("name")
        if name == "base_link" or name.startswith(HAND_PREFIX) or len(link):
            continue
        incoming = next(j for j in robot.findall("joint") if j.find("child").get("link") == name)
        if incoming.get("type") != "fixed":
            raise ValueError(f"Cannot remove a moving empty frame: {name}")
        parent = incoming.find("parent").get("link")
        transform = origin_matrix(incoming.find("origin"))
        for joint in robot.findall("joint"):
            if joint.find("parent").get("link") == name:
                joint.find("parent").set("link", parent)
                _set_origin(joint, transform @ origin_matrix(joint.find("origin")))
        robot.remove(incoming)
        robot.remove(link)
    # UR puts the physical base in a fixed rotated child of an empty base_link.
    root = robot.find("link[@name='base_link']")
    physical = robot.find("link[@name='base_link_inertia']")
    if root is not None and physical is not None and len(root) == 0:
        incoming = next(j for j in robot.findall("joint")
                        if j.find("child").get("link") == "base_link_inertia")
        transform = origin_matrix(incoming.find("origin"))
        for element in physical:
            element = copy.deepcopy(element)
            _set_origin(element, transform @ origin_matrix(element.find("origin")))
            root.append(element)
        for joint in robot.findall("joint"):
            if joint.find("parent").get("link") == "base_link_inertia":
                joint.find("parent").set("link", "base_link")
                _set_origin(joint, transform @ origin_matrix(joint.find("origin")))
        robot.remove(incoming)
        robot.remove(physical)


def expand_arm(description: Path) -> ET.Element:
    """Expand local UR xacro without requiring ROS package discovery or a ROS bridge."""
    import xacro

    with tempfile.TemporaryDirectory(prefix="ur5e-tactile-xacro-") as scratch:
        scratch = Path(scratch)
        # Resolve package substitution in a temporary copy, including nested includes.
        for source in (description / "urdf").rglob("*.xacro"):
            dest = scratch / source.relative_to(description)
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_text().replace(
                "$(find ur_description)/urdf", str(scratch / "urdf")
            ).replace("$(find ur_description)", str(description))
            dest.write_text(content)
        wrapper = ET.Element("robot", {"name": "ur5e_inspire", "xmlns:xacro": "http://www.ros.org/wiki/xacro"})
        ET.SubElement(wrapper, "xacro:include", {"filename": str(scratch / "urdf/ur_macro.xacro")})
        ET.SubElement(wrapper, "link", {"name": "world"})
        macro = ET.SubElement(wrapper, "xacro:ur_robot", {
            "name": "ur5e", "tf_prefix": "", "parent": "world", "ur_type": "ur5e",
            "generate_ros2_control_tag": "false",
            "joint_limits_parameters_file": str(description / "config/ur5e/joint_limits.yaml"),
            "kinematics_parameters_file": str(description / "config/ur5e/default_kinematics.yaml"),
            "physical_parameters_file": str(description / "config/ur5e/physical_parameters.yaml"),
            "visual_parameters_file": str(description / "config/ur5e/visual_parameters.yaml"),
        })
        ET.SubElement(macro, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        entry = scratch / "robot.xacro"
        ET.ElementTree(wrapper).write(entry, encoding="unicode")
        result = ET.fromstring(xacro.process_file(str(entry)).toxml())
    for element in list(result):
        if element.tag == "link" and element.get("name") == "world":
            result.remove(element)
        elif element.tag == "joint" and element.find("parent").get("link") == "world":
            result.remove(element)
        elif element.tag not in ("link", "joint", "material"):
            result.remove(element)
    for mesh in result.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("package://ur_description/", str(description) + "/"))
    return result


def build_model(root: Path, output: Path) -> tuple[Path, dict]:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hand_path = root / "src/inspire_robot/urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf"
    hand = ET.parse(hand_path).getroot()
    arm = expand_arm(root / "src/Universal_Robots_ROS2_Description")
    # Both tool0 +Z and the hand's base +Z are the outward mounting axis.
    # Keep these coaxial; obtain the grasp orientation with ARM_POSE, not a
    # 90-degree bend in this fixed joint. Both mounting-face origins coincide;
    # do not leave the old unmodeled 25 mm spacer as a visible air gap.
    mount_rpy = np.zeros(3)
    mount = ET.SubElement(arm, "joint", {"name": "inspire_mount", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "tool0"})
    ET.SubElement(mount, "child", {"link": HAND_PREFIX + "base_link"})
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": " ".join(map(str, mount_rpy))})
    for source in hand:
        if source.tag not in ("link", "joint", "material"):
            continue
        element = copy.deepcopy(source)
        if "name" in element.attrib:
            element.set("name", HAND_PREFIX + element.get("name"))
        for sub in element.iter():
            for attr in ("link", "joint"):
                if attr in sub.attrib:
                    sub.set(attr, HAND_PREFIX + sub.get(attr))
            if sub.tag == "material" and "name" in sub.attrib and sub is not element:
                sub.set("name", HAND_PREFIX + sub.get("name"))
        for mesh in element.iter("mesh"):
            # Both original ROS URI and the earlier relative-path variant are accepted.
            mesh.set("filename", str(hand_path.parent.parent / "meshes" / Path(mesh.get("filename")).name))
        # Mimic is implemented by hand_targets(), not simultaneously by PhysX
        # constraints AND independent drives. Keep ratios in the source only.
        for mimic in element.findall("mimic"):
            element.remove(mimic)
        arm.append(element)
    sensor_names = [HAND_PREFIX + name for name in SENSOR_NAMES]
    actual_sensors = {l.get("name") for l in arm.findall("link") if "force_sensor" in l.get("name")}
    if actual_sensors != set(sensor_names):
        raise ValueError(f"Unexpected sensor links: {actual_sensors.symmetric_difference(sensor_names)}")
    for mesh in arm.iter("mesh"):
        if not Path(mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))
    positions = ARM_POSE | hand_targets(hand, OPEN)
    hand_pose = forward_kinematics(arm, positions)[HAND_PREFIX + "base_link"]
    collapse_empty_arm_frames(arm)
    if not np.allclose(forward_kinematics(arm, positions)[HAND_PREFIX + "base_link"], hand_pose):
        raise RuntimeError("Frame simplification changed the hand mount")
    metadata = {
        "sensor_names": sensor_names, "arm_pose": ARM_POSE,
        "open_targets": hand_targets(hand, OPEN), "closed_targets": hand_targets(hand, CLOSED),
        "hand_pose": hand_pose.tolist(), "mount_rpy": mount_rpy.tolist(),
        "source_hand": str(hand_path),
    }
    ET.indent(arm, space="  ")
    path = output / "ur5e_inspire.urdf"
    content = ET.tostring(arm, encoding="unicode")
    # Keep the USD converter cache valid on repeated unchanged runs.
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path, metadata
