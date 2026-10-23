"""Extend Holding v0's initial-obstacle capacity without changing its task."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v0.env_cfg import FusionHoldingV0EnvCfg


@configclass
class FusionHoldingV1EnvCfg(FusionHoldingV0EnvCfg):
    # Keep the inherited 5 Gray + 5 White scene and population curriculum.
    # The extra slots support up to sixteen obstacles and are zero when unused.
    obstacle_observation_slots: int = 16
