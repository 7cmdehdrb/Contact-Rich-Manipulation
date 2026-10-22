"""Extend Fusion v3 with explicit, independent manager configurations."""

import math

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v3.env_cfg import ContactSweepClutterFusionV3EnvCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_0CurriculumCfg
from .events_cfg import ContactSweepClutterFusionV1_0EventsCfg
from .rewards_cfg import ContactSweepClutterFusionV1_0RewardsCfg
from .scene_cfg import ContactSweepClutterFusionV1_0SceneCfg
from .terminations_cfg import ContactSweepClutterFusionV1_0TerminationsCfg


@configclass
class ContactSweepClutterFusionV1_0EnvCfg(ContactSweepClutterFusionV3EnvCfg):
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    scene: ContactSweepClutterFusionV1_0SceneCfg = ContactSweepClutterFusionV1_0SceneCfg(
        num_envs=4096, env_spacing=2.5,
    )
    rewards: ContactSweepClutterFusionV1_0RewardsCfg = ContactSweepClutterFusionV1_0RewardsCfg()
    terminations: ContactSweepClutterFusionV1_0TerminationsCfg = ContactSweepClutterFusionV1_0TerminationsCfg()
    curriculum: ContactSweepClutterFusionV1_0CurriculumCfg = ContactSweepClutterFusionV1_0CurriculumCfg()
    events: ContactSweepClutterFusionV1_0EventsCfg = ContactSweepClutterFusionV1_0EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.configure_population_capacity(self.max_gray_obstacles, self.max_white_obstacles)
        self.events.object_spawn.params["gray_obstacles"] = min(1, self.max_gray_obstacles)
        self.events.object_spawn.params["white_obstacles"] = min(1, self.max_white_obstacles)
        # Also correct before the first runner update/reset.
        self.events.object_spawn.params["excluded_x_half_angle"] = math.radians(30.0)
