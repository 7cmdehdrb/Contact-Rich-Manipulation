"""Override placement and physics while retaining the parent's reset ordering."""

from isaaclab.utils import configclass

from ..contact_sweep.events_cfg import ContactSweepEventsCfg
from .events import randomize_clutter_scene
from .physics import initialize_clutter_materials, randomize_clutter_episode_physics


@configclass
class ContactSweepClutterEventsCfg(ContactSweepEventsCfg):
    def __post_init__(self):
        self.object_spawn.func = randomize_clutter_scene
        self.object_spawn.params["observation_noise"] = 0.01
        self.physics_materials.func = initialize_clutter_materials
        self.physics_randomization.func = randomize_clutter_episode_physics
