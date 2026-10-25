"""Test randomized-object learning with Holding v2's joint-position control.

Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props.

Holding OSC v2 with randomized props growing from 1/1 to 8/8."""

from ..assets.scenes.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomSceneCfg
from ..assets.scenes.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomV1SceneCfg
from ..mdp.curricula.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomCurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomV1CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomV2CurriculumCfg
from ..mdp.events.contact_sweep_clutter_fusion_holding_osc_object_random import ObjectRandomEventsCfg
from .contact_sweep_clutter_fusion_holding_envs import FusionHoldingV2EnvCfg
from .contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV0EnvCfg
from .contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV2EnvCfg
from .contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV3EnvCfg
from .contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV4EnvCfg
from isaaclab.utils import configclass

@configclass
class FusionHoldingObjectRandomTestV0EnvCfg(FusionHoldingV2EnvCfg):
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV1CurriculumCfg = ObjectRandomV1CurriculumCfg()


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


@configclass
class FusionHoldingOSCObjectRandomV1EnvCfg(FusionHoldingOSCV2EnvCfg):
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV1CurriculumCfg = ObjectRandomV1CurriculumCfg()


@configclass
class FusionHoldingOSCObjectRandomV2EnvCfg(FusionHoldingOSCV3EnvCfg):
    # Keep the physical slot layout; curriculum limits active counts to 6/6.
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV2CurriculumCfg = ObjectRandomV2CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # TEMP SUCCESS DEBUG (disabled): preserved with the commented print block.
        # self.terminations.success.params["debug_print"] = True


@configclass
class FusionHoldingOSCObjectRandomV3EnvCfg(FusionHoldingOSCV4EnvCfg):
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV2CurriculumCfg = ObjectRandomV2CurriculumCfg()
