"""Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props.

Holding OSC v2 with randomized props growing from 1/1 to 8/8."""

from ..contact_sweep_clutter_fusion_holding_osc_object_random_spawners import prop_variants
from .contact_sweep_clutter_fusion import ContactSweepClutterFusionSceneCfg
from .contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0SceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim

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
class ObjectRandomV1SceneCfg(ObjectRandomSceneCfg):
    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=16):
        super().configure_population_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
