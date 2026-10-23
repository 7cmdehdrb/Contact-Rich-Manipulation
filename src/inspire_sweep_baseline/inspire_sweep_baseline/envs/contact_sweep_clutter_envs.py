"""Contact Sweep child with three physical obstacles and a learned scene embedding.

Clutter v0 inheritance with asymmetric observations and four added rewards.

Clutter v1 plus four reward curricula.

Preserve the checkpoint's policy, rewards, curricula and physics contracts.

Clutter v1 with a compensated six-axis observation and one new cost."""

from ..assets.scenes.contact_sweep_clutter import ContactSweepClutterSceneCfg
from ..mdp.curricula.contact_sweep_clutter import ContactSweepClutterV1CurriculumCfg
from ..mdp.events.contact_sweep_clutter import ContactSweepClutterEventsCfg
from ..mdp.observations.contact_sweep_clutter import ContactSweepClutterObservationsCfg
from ..mdp.observations.contact_sweep_clutter import ContactSweepClutterV1ObservationsCfg
from ..mdp.observations.contact_sweep_clutter import ContactSweepClutterV2ObservationsCfg
from ..mdp.rewards.contact_sweep_clutter import ContactSweepClutterRewardsCfg
from ..mdp.rewards.contact_sweep_clutter import ContactSweepClutterV1CurriculumRewardsCfg
from ..mdp.rewards.contact_sweep_clutter import ContactSweepClutterV1RewardsCfg
from ..mdp.rewards.contact_sweep_clutter import ContactSweepClutterV2RewardsCfg
from ..mdp.terminations.contact_sweep_clutter import ContactSweepClutterTerminationsCfg
from .contact_sweep_envs import ContactSweepEnvCfg
from isaaclab.utils import configclass

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


@configclass
class ContactSweepClutterV1EnvCfg(ContactSweepClutterEnvCfg):
    observations: ContactSweepClutterV1ObservationsCfg = ContactSweepClutterV1ObservationsCfg()
    rewards: ContactSweepClutterV1RewardsCfg = ContactSweepClutterV1RewardsCfg()


@configclass
class ContactSweepClutterV1CurriculumEnvCfg(ContactSweepClutterV1EnvCfg):
    rewards: ContactSweepClutterV1CurriculumRewardsCfg = ContactSweepClutterV1CurriculumRewardsCfg()
    curriculum: ContactSweepClutterV1CurriculumCfg = ContactSweepClutterV1CurriculumCfg()


@configclass
class ContactSweepClutterFTTestEnvCfg(ContactSweepClutterV1CurriculumEnvCfg):
    """The v2 compensation runs outside policy observations and rewards."""


@configclass
class ContactSweepClutterV2EnvCfg(ContactSweepClutterV1EnvCfg):
    observations: ContactSweepClutterV2ObservationsCfg = ContactSweepClutterV2ObservationsCfg()
    rewards: ContactSweepClutterV2RewardsCfg = ContactSweepClutterV2RewardsCfg()
