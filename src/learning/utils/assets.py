"""Project asset path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.learning.utils.config import repo_root


@dataclass(frozen=True)
class UR5eDescriptionPaths:
    root: Path
    urdf_xacro: Path
    config_dir: Path
    mesh_dir: Path


def ur5e_description_paths() -> UR5eDescriptionPaths:
    root = repo_root() / "src" / "Universal_Robots_ROS2_Description"
    return UR5eDescriptionPaths(
        root=root,
        urdf_xacro=root / "urdf" / "ur.urdf.xacro",
        config_dir=root / "config" / "ur5e",
        mesh_dir=root / "meshes" / "ur5e",
    )


def validate_ur5e_description() -> UR5eDescriptionPaths:
    paths = ur5e_description_paths()
    missing = [path for path in (paths.root, paths.urdf_xacro, paths.config_dir, paths.mesh_dir) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"UR5e description files are missing: {formatted}")
    return paths
