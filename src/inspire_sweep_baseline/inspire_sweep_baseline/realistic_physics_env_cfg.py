"""Inherited baseline experiment varying only object mass and contact friction."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .env_cfg import EventsCfg, InspireSweepEnvCfg
from .mdp.realistic_physics import MASS_MEAN, initialize_realistic_materials, randomize_episode_physics


@configclass
class RealisticPhysicsEventsCfg(EventsCfg):
    # Bind reusable materials once; sample again on EVERY episode reset.
    physics_materials = EventTerm(func=initialize_realistic_materials, mode="startup")
    physics_randomization = EventTerm(func=randomize_episode_physics, mode="reset")


@configclass
class InspireSweepRealisticPhysicsEnvCfg(InspireSweepEnvCfg):
    events: RealisticPhysicsEventsCfg = RealisticPhysicsEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        cylinder = self.scene.object_collection.rigid_objects["mass_cylinder"]
        cylinder.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=MASS_MEAN)
