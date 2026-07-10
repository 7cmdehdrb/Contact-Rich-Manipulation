"""Generate a flattened UR5e URDF from the vendored xacro description."""

from __future__ import annotations

import argparse
import fcntl
import shutil
from pathlib import Path

import xacro

from src.learning.utils.assets import ur5e_description_paths
from src.learning.utils.config import repo_root


def generate_ur5e_urdf(output_path: Path | None = None) -> Path:
    paths = ur5e_description_paths()
    if output_path is None:
        output_path = repo_root() / "src" / "learning" / "generated" / "ur5e.urdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = output_path.parent / ".ur5e_urdf.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _is_generated_urdf_ready(output_path, paths.root):
            return output_path

        patched_root = output_path.parent / "ur_description"
        if patched_root.exists():
            shutil.rmtree(patched_root)
        shutil.copytree(paths.root / "urdf", patched_root / "urdf")
        (patched_root / "config").symlink_to(paths.root / "config", target_is_directory=True)
        (patched_root / "meshes").symlink_to(paths.root / "meshes", target_is_directory=True)

        for xacro_file in patched_root.rglob("*.xacro"):
            text = xacro_file.read_text(encoding="utf-8")
            text = text.replace("$(find ur_description)", str(patched_root))
            xacro_file.write_text(text, encoding="utf-8")

        top_level = patched_root / "urdf" / "ur.urdf.xacro"
        doc = xacro.process_file(
            str(top_level),
            mappings={
                "name": "ur5e",
                "ur_type": "ur5e",
                "force_abs_paths": "true",
            },
        )
        xml = doc.toprettyxml(indent="  ")
        xml = xml.replace("file:///opt/ros/humble/share/ur_description", str(paths.root))
        xml = xml.replace(f"file://{patched_root}", str(patched_root))
        xml = xml.replace("file://", "")
        output_path.write_text(xml, encoding="utf-8")
    return output_path


def _is_generated_urdf_ready(output_path: Path, ur_description_root: Path) -> bool:
    if not output_path.exists():
        return False
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    return str(ur_description_root / "meshes" / "ur5e") in text and "tool0" in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "src" / "learning" / "generated" / "ur5e.urdf",
    )
    args = parser.parse_args()
    path = generate_ur5e_urdf(args.output)
    print(path)


if __name__ == "__main__":
    main()
