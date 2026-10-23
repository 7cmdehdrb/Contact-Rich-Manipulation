"""Inherited baseline experiment varying only object mass and contact friction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...mdp.common.realistic_physics import MASS_MEAN
from ..realistic_spawners import RealisticCylinderSpawnerCfg
from ..realistic_spawners import RealisticRobotSpawnerCfg
from ..realistic_spawners import RealisticShelfSpawnerCfg
from ..realistic_spawners import replace_friction_spawner
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from ...envs.sweep_envs import InspireSweepEnvCfg


def apply_realistic_physics(cfg: "InspireSweepEnvCfg"):
    """Apply the existing mass and spawn-time friction setup to a baseline cfg.

    Call after the baseline ``__post_init__`` and use
    ``RealisticPhysicsEventsCfg`` (or a subclass) for episode randomization.
    """
    # No prestartup USD randomization: retain Baseline physics replication.
    cylinder = cfg.scene.object_collection.rigid_objects["mass_cylinder"]
    cylinder.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=MASS_MEAN)
    defaults = cfg.sim.physics_material
    cfg.scene.robot.spawn = replace_friction_spawner(cfg.scene.robot.spawn, RealisticRobotSpawnerCfg, defaults)
    cfg.scene.shelf.spawn = replace_friction_spawner(cfg.scene.shelf.spawn, RealisticShelfSpawnerCfg, defaults)
    cylinder.spawn = replace_friction_spawner(cylinder.spawn, RealisticCylinderSpawnerCfg, defaults)
