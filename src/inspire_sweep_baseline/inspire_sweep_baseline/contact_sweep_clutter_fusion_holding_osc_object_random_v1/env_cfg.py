"""Holding OSC v2 with the existing fourteen-prop randomization contract."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_osc_v2.env_cfg import FusionHoldingOSCV2EnvCfg
from ..contact_sweep_clutter_fusion_holding_v2.curriculum_cfg import FusionHoldingV2CurriculumCfg
from ..contact_sweep_clutter_fusion_holding_osc_object_random_v0.env_cfg import (
    ObjectRandomEventsCfg,
    ObjectRandomSceneCfg,
)


@configclass
class ObjectRandomV1SceneCfg(ObjectRandomSceneCfg):
    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=16):
        super().configure_population_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)


@configclass
class ObjectRandomV1CurriculumCfg(FusionHoldingV2CurriculumCfg):
    obstacle_population = None  # Preserve the fixed fourteen-obstacle scene.


@configclass
class FusionHoldingOSCObjectRandomV1EnvCfg(FusionHoldingOSCV2EnvCfg):
    max_gray_obstacles: int = 7
    max_white_obstacles: int = 7
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV1CurriculumCfg = ObjectRandomV1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params.update(
            gray_obstacles=self.max_gray_obstacles,
            white_obstacles=self.max_white_obstacles,
        )
