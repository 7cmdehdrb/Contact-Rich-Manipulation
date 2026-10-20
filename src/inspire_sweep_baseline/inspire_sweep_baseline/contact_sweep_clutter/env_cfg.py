"""Contact Sweep child with three physical obstacles and a learned scene embedding."""

from isaaclab.utils import configclass

from ..contact_sweep.env_cfg import ContactSweepEnvCfg
from .events_cfg import ContactSweepClutterEventsCfg
from .observations_cfg import ContactSweepClutterObservationsCfg
from .rewards_cfg import ContactSweepClutterRewardsCfg
from .scene_cfg import ContactSweepClutterSceneCfg
from .terminations_cfg import ContactSweepClutterTerminationsCfg


@configclass
class ContactSweepClutterEnvCfg(ContactSweepEnvCfg):
    scene: ContactSweepClutterSceneCfg = ContactSweepClutterSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ContactSweepClutterObservationsCfg = ContactSweepClutterObservationsCfg()
    events: ContactSweepClutterEventsCfg = ContactSweepClutterEventsCfg()
    rewards: ContactSweepClutterRewardsCfg = ContactSweepClutterRewardsCfg()
    terminations: ContactSweepClutterTerminationsCfg = ContactSweepClutterTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for obj in self.scene.object_collection.rigid_objects.values():
            obj.spawn.friction_defaults = self.sim.physics_material.copy()
