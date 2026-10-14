"""High-friction scene variant used only by the tilted cantilever test."""

from __future__ import annotations

from pathlib import Path

from isaaclab.scene import InteractiveSceneCfg

from .scene import make_scene
from .tilted_cantilever_model import (
    TILTED_DYNAMIC_FRICTION,
    TILTED_STATIC_FRICTION,
)


def make_tilted_scene(
    urdf_path: Path,
    model: dict[str, object],
    *,
    num_envs: int,
    env_spacing: float,
    force_conversion: bool,
) -> InteractiveSceneCfg:
    """Reuse the base scene while increasing payload/tool contact friction."""
    cfg = make_scene(
        urdf_path,
        model,
        num_envs=num_envs,
        env_spacing=env_spacing,
        force_conversion=force_conversion,
    )
    material = cfg.payload.spawn.physics_material
    material.static_friction = TILTED_STATIC_FRICTION
    material.dynamic_friction = TILTED_DYNAMIC_FRICTION
    material.friction_combine_mode = "max"
    return cfg

