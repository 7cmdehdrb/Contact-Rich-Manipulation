"""Tactile-localization copy of the wide-randomization sweep task.

The policy cannot observe the target object's current pose. The simulator may
still use that state for rewards and terminations as privileged training data.
"""

from isaaclab.utils import configclass

from .env_cfg_wide_randomization import UR5eOscSweepWideRandomizationEnvCfg


@configclass
class UR5eOscSweepTactileLocalizationEnvCfg(UR5eOscSweepWideRandomizationEnvCfg):
    """Wide-randomization task without current target pose observations."""

    def __post_init__(self):
        super().__post_init__()

        # Remove only the real-time target pose from the policy observation.
        # Rewards and terminations continue to access target_object directly.
        self.observations.policy.current_target_pose = None

