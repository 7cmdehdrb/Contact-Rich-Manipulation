"""Inherited baseline experiment varying only object mass and contact friction."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .env_cfg import EventsCfg, InspireSweepEnvCfg
from .assets.realistic_spawners import (
    RealisticRobotSpawnerCfg, RealisticShelfSpawnerCfg, RealisticCylinderSpawnerCfg, replace_friction_spawner,
)
from .mdp.realistic_physics import MASS_MEAN, initialize_realistic_materials, randomize_episode_physics


def apply_realistic_physics(cfg: InspireSweepEnvCfg):
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


@configclass
class RealisticPhysicsEventsCfg(EventsCfg):
    # Source spawners already bind Average materials before physics cloning.
    # Cache Tensor API buffers and bounded Gaussian material pools once.
    physics_materials = EventTerm(func=initialize_realistic_materials, mode="startup")
    physics_randomization = EventTerm(func=randomize_episode_physics, mode="reset")


@configclass
class InspireSweepRealisticPhysicsEnvCfg(InspireSweepEnvCfg):
    events: RealisticPhysicsEventsCfg = RealisticPhysicsEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_realistic_physics(self)
