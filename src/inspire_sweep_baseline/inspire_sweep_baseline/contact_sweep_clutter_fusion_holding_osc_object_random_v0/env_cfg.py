"""Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props."""
import torch
import isaaclab.sim as sim
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_holding_osc_v0.env_cfg import FusionHoldingOSCV0EnvCfg
from ..contact_sweep_clutter_fusion_v1_0.scene_cfg import ContactSweepClutterFusionV1_0SceneCfg
from ..contact_sweep_clutter_fusion_v1_0.events_cfg import ContactSweepClutterFusionV1_0EventsCfg
from ..contact_sweep_clutter.physics import randomize_clutter_episode_physics
from ..contact_sweep_clutter_fusion.scene_cfg import ContactSweepClutterFusionSceneCfg
from .spawn import prop_variants
from .events import randomize_large_prop_scene
from ..contact_sweep_clutter_fusion_holding_v0.curriculum_cfg import FusionHoldingV0CurriculumCfg


def randomize_prop_physics(env, env_ids, mass_scale=1.5):
    randomize_clutter_episode_physics(env,env_ids,mass_scale)
    # Same sampled masses/friction, but preserve each shape's own inertia/mass.
    ids=torch.arange(env.num_envs) if env_ids is None else torch.as_tensor(env_ids,device='cpu',dtype=torch.long)
    state=env._clutter_physics
    objects=env.scene['object_collection']
    rows=(torch.arange(objects.num_objects)[:,None]*env.num_envs+ids[None,:]).flatten()
    state.inertia_buffer[rows]=state.default_inertia[rows]*(state.mass_buffer[rows]/state.default_mass[rows])
    objects.root_physx_view.set_inertias(state.inertia_buffer,rows)


@configclass
class ObjectRandomSceneCfg(ContactSweepClutterFusionV1_0SceneCfg):
    replicate_physics: bool = False

    def configure_population_capacity(self,max_gray_obstacles,max_white_obstacles,observation_slots=14):
        super().configure_population_capacity(max_gray_obstacles,max_white_obstacles, observation_slots=observation_slots)
        original=ContactSweepClutterFusionSceneCfg().object_collection.rigid_objects['mass_cylinder'].spawn
        for name, obj in self.object_collection.rigid_objects.items():
            if name == 'mass_cylinder':
                continue
            obj.spawn=sim.MultiAssetSpawnerCfg(assets_cfg=prop_variants(original),random_choice=True,activate_contact_sensors=True)


@configclass
class ObjectRandomEventsCfg(ContactSweepClutterFusionV1_0EventsCfg):
    object_spawn=ContactSweepClutterFusionV1_0EventsCfg().object_spawn.replace(
        func=randomize_large_prop_scene,
        params={**ContactSweepClutterFusionV1_0EventsCfg().object_spawn.params, 'object_radius': .04})
    physics_randomization=ContactSweepClutterFusionV1_0EventsCfg().physics_randomization.replace(func=randomize_prop_physics)


@configclass
class ObjectRandomCurriculumCfg(FusionHoldingV0CurriculumCfg):
    obstacle_population = None  # Temporary max-population test mode.


@configclass
class FusionHoldingOSCObjectRandomV0EnvCfg(FusionHoldingOSCV0EnvCfg):
    max_gray_obstacles: int = 7
    max_white_obstacles: int = 7
    obstacle_observation_slots: int = 14
    scene: ObjectRandomSceneCfg = ObjectRandomSceneCfg(num_envs=4096,env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()

    curriculum: ObjectRandomCurriculumCfg = ObjectRandomCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params.update(gray_obstacles=self.max_gray_obstacles,
                                               white_obstacles=self.max_white_obstacles)
