"""Override placement and physics while retaining the parent's reset ordering."""

from ..event_functions.contact_sweep_clutter import randomize_clutter_scene
from ..event_functions.contact_sweep_clutter_physics import initialize_clutter_materials
from ..event_functions.contact_sweep_clutter_physics import randomize_clutter_episode_physics
from .contact_sweep import ContactSweepEventsCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterEventsCfg(ContactSweepEventsCfg):
    def __post_init__(self):
        self.object_spawn.func = randomize_clutter_scene
        self.object_spawn.params["observation_noise"] = 0.01
        self.physics_materials.func = initialize_clutter_materials
        self.physics_randomization.func = randomize_clutter_episode_physics
