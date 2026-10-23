"""Holding OSC v1 with the existing fourteen-prop randomization contract."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_osc_v1.env_cfg import FusionHoldingOSCV1EnvCfg
from ..contact_sweep_clutter_fusion_holding_osc_object_random_v0.env_cfg import (
    ObjectRandomCurriculumCfg,
    ObjectRandomEventsCfg,
    ObjectRandomSceneCfg,
)


@configclass
class ObjectRandomV1SceneCfg(ObjectRandomSceneCfg):
    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=16):
        super().configure_population_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)


@configclass
class FusionHoldingOSCObjectRandomV1EnvCfg(FusionHoldingOSCV1EnvCfg):
    max_gray_obstacles: int = 7
    max_white_obstacles: int = 7
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomCurriculumCfg = ObjectRandomCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params.update(
            gray_obstacles=self.max_gray_obstacles,
            white_obstacles=self.max_white_obstacles,
        )
