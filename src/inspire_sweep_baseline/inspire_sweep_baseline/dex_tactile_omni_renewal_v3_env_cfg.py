"""Dex Tactile v1 with heavier objects and empirically balanced progress rewards."""

from isaaclab.utils import configclass

from .dex_tactile_omni_renewal_v1_env_cfg import InspireDexTactileSweepOmniRenewalV1EnvCfg
from .mdp.realistic_physics import randomize_scaled_episode_physics


@configclass
class InspireDexTactileSweepOmniRenewalV3EnvCfg(InspireDexTactileSweepOmniRenewalV1EnvCfg):
    """Branch directly from v1; calibrate progress weights against its logged returns."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pushing_speed.params["free_speed"] = 0.2
        self.rewards.pushing_speed.weight = -2.0
        # v1 run 2026-10-19_12-15-47, iterations 39027–39036:
        # 300 * (4/3) * 0.7783182442 / 3.3145449400 = 93.9276.
        # Rounded to 94: reaching:goal contribution approximately 3:4.
        self.rewards.goal_distance.weight = 94.0
        cylinder = self.scene.object_collection.rigid_objects["mass_cylinder"]
        cylinder.spawn.mass_props.mass *= 1.5
        # Scale the reset samples as well as the spawn mass. The shared event
        # updates inertia consistently from cached defaults, without compounding.
        self.events.physics_randomization.func = randomize_scaled_episode_physics
        self.events.physics_randomization.params["mass_scale"] = 1.5
