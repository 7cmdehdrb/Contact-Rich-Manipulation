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

        # The previous OSC clamp used only 90% of the UR5e rating.  Heavy
        # randomized objects (up to 3 kg) could therefore saturate before they
        # started sliding.  Raise both the simulated actuator limit and the OSC
        # clamp together; changing only one would leave the other as the actual
        # bottleneck.  The wrist F/T safety termination remains active.
        arm_actuator = self.scene.robot.actuators["arm"]
        arm_actuator.effort_limit_sim = {
            joint_name: 1.5 * limit
            for joint_name, limit in arm_actuator.effort_limit_sim.items()
        }
        self.actions.arm_action.effort_limit_scale = 1.0

        # Make endpoint placement dominate the attractive but incomplete local
        # optimum of centered contact plus accurate force.  A wide Gaussian is
        # informative from the initial 10--22 cm error, while the original
        # narrow Gaussian still rewards precise final placement.
        self.rewards.normalized_progress.weight = 3.0
        self.rewards.endpoint_tracking.weight = 12.0
        self.rewards.endpoint_tracking.params["coarse_std"] = 0.12
        self.rewards.endpoint_tracking.params["coarse_weight"] = 0.35
        self.rewards.success.weight = 40.0

