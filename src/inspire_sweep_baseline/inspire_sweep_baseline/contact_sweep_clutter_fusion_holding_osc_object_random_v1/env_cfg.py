"""Holding OSC v2 with randomized props growing from 1/1 to 8/8."""

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
    obstacle_population = FusionHoldingV2CurriculumCfg().obstacle_population.replace(
        params={**FusionHoldingV2CurriculumCfg().obstacle_population.params,
                "final_gray": 8, "final_white": 8},
    )


@configclass
class FusionHoldingOSCObjectRandomV1EnvCfg(FusionHoldingOSCV2EnvCfg):
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV1CurriculumCfg = ObjectRandomV1CurriculumCfg()

    # Inherit the parent's 1 Gray + 1 White initial reset counts. The curriculum
    # changes future resets without changing the current episode's active mask.
