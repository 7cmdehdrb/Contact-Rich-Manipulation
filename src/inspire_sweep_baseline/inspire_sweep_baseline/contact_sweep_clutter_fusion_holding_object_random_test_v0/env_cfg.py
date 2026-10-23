"""Test randomized-object learning with Holding v2's joint-position control."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v2.env_cfg import FusionHoldingV2EnvCfg
from ..contact_sweep_clutter_fusion_holding_osc_object_random_v1.env_cfg import (
    ObjectRandomEventsCfg,
    ObjectRandomV1CurriculumCfg,
    ObjectRandomV1SceneCfg,
)


@configclass
class FusionHoldingObjectRandomTestV0EnvCfg(FusionHoldingV2EnvCfg):
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8
    scene: ObjectRandomV1SceneCfg = ObjectRandomV1SceneCfg(num_envs=4096, env_spacing=2.5)
    events: ObjectRandomEventsCfg = ObjectRandomEventsCfg()
    curriculum: ObjectRandomV1CurriculumCfg = ObjectRandomV1CurriculumCfg()

    # Inherit Holding v2 actions and actuator PD unchanged. Reusing randomized
    # scene/manager configs does not run the OSC environment's post-init.
