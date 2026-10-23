"""Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props."""

from ..event_functions.contact_sweep_clutter_fusion_holding_osc_object_random import randomize_large_prop_scene
from ..event_functions.contact_sweep_clutter_fusion_holding_osc_object_random import randomize_prop_physics
from .contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0EventsCfg
from isaaclab.utils import configclass

@configclass
class ObjectRandomEventsCfg(ContactSweepClutterFusionV1_0EventsCfg):
    object_spawn=ContactSweepClutterFusionV1_0EventsCfg().object_spawn.replace(
        func=randomize_large_prop_scene,
        params={**ContactSweepClutterFusionV1_0EventsCfg().object_spawn.params, 'object_radius': .04})
    physics_randomization=ContactSweepClutterFusionV1_0EventsCfg().physics_randomization.replace(func=randomize_prop_physics)
