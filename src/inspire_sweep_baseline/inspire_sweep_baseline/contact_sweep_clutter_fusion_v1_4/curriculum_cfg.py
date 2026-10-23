"""Keep v1.0 deadzone/population timing; calibrate thresholds at 250 iterations."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v1_0.curriculum_cfg import ContactSweepClutterFusionV1_0CurriculumCfg
from .curriculum import staged_reward_threshold


@configclass
class ContactSweepClutterFusionV1_4CurriculumCfg(ContactSweepClutterFusionV1_0CurriculumCfg):
    obstacle_population = ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.replace(
        params={**ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.params,
                "final_gray": 5, "final_white": 5},
    )
    # Preserve the parent's first decrement at 500 and decrement per iteration.
    # Endpoints are explicit because 3300/13300/1900 are off the 250-step grid.
    arm_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "arm_speed_excess", "parameter": "free_speed",
                "initial": 1.0, "stages": ((500, 3300, 0.98, 0.7),), "interval": 250},
    )
    eef_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "eef_linear_speed", "parameter": "free_speed",
                "initial": 1.5, "stages": ((500, 13300, 1.48, 0.2),), "interval": 250},
    )
    pushing_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "pushing_speed", "parameter": "free_speed",
                "initial": 0.1, "stages": ((500, 1900, 0.09, 0.02),), "interval": 250},
    )
    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "object_acceleration", "parameter": "free_acceleration",
                "initial": 100.0, "stages": ((10000, 10000, 50.0, 50.0),
                                            (12000, 18000, 50.0, 10.0)), "interval": 250},
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "wrench_change", "parameter": "free_force_change",
                "initial": 5000.0, "stages": ((10000, 10000, 200.0, 200.0),
                                             (12000, 18000, 200.0, 100.0)), "interval": 250},
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "wrench_change", "parameter": "free_moment_change",
                "initial": 500.0, "stages": ((10000, 10000, 30.0, 30.0),
                                            (12000, 18000, 30.0, 20.0)), "interval": 250},
    )
    goal_hold_tolerance = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={"term_name": "goal_hold", "parameter": "position_tolerance",
                "initial": 0.0, "stages": ((5000, 10000, 0.01, 0.005),), "interval": 250},
    )
