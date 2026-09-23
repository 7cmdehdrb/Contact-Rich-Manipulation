"""Read-only virtual sensing regions; NEVER used to author physics geometry."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from .model import origin_matrix


def stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little") if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        triangles = np.frombuffer(data, dtype=np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84)
        return triangles["vertices"].reshape(-1, 3).astype(np.float64)
    vertices = [line.split()[1:] for line in data.decode("ascii").splitlines()
                if line.strip().startswith("vertex ")]
    return np.asarray(vertices, dtype=np.float64)


def load_regions(urdf: Path, names: list[str]) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Parent link -> (channel index, sensor-local convex halfspaces).

    Only the existing mesh is read. Offsetting these planes in the reader
    affects classification tolerance, not collision shapes or contact offsets.
    """
    root = ET.parse(urdf).getroot()
    result = {}
    for index, name in enumerate(names):
        link = root.find(f"link[@name='{name}']")
        joint = next(j for j in root.findall("joint") if j.find("child").get("link") == name)
        if joint.get("type") != "fixed":
            raise ValueError(f"Region must have a fixed parent: {name}")
        collisions = link.findall("collision")
        if len(collisions) != 1 or collisions[0].find("geometry/mesh") is None:
            raise ValueError(f"Expected one sensor collision mesh: {name}")
        collision = collisions[0]
        mesh = collision.find("geometry/mesh")
        path = Path(mesh.get("filename"))
        if not path.is_absolute():
            path = urdf.parent / path
        vertices = stl_vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        transform = origin_matrix(collision.find("origin"))
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        # Round duplicate triangle planes to reduce per-step classification work.
        equations = np.unique(np.round(ConvexHull(vertices).equations, 10), axis=0)
        result.setdefault(joint.find("parent").get("link"), []).append((index, equations))
    return result


def observer_key(parent: str) -> str:
    return "contact_region_" + parent
