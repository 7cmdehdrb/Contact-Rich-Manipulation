"""Five absolute schedules in the runner's Learning iteration units.

Absolute iteration schedules; no inherited terms or tactile schedules.

Absolute iteration schedules; independent terms with goal and contact-width schedules.

Keep v1.0 deadzone/population timing; calibrate thresholds at 250 iterations.

Fusion v3 schedules in the runner's zero-based Learning iteration units."""

from ..curriculum_functions.contact_sweep_clutter import reward_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import delayed_reward_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import goal_deadzone_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import obstacle_population_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import piecewise_linear_reward_threshold
from ..curriculum_functions.contact_sweep_clutter_fusion import staged_reward_threshold
from ..curriculum_functions.contact_sweep_clutter_fusion_quantized_ramps import staged_reward_threshold as staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterFusionCurriculumCfg:
    pushing_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "decrement": 0.01,
            "interval": 300,
            "minimum": 0.05,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "decrement": 0.02,
            "interval": 300,
            "minimum": 0.8,
        },
    )
    action_rate_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "action_rate",
            "start": -0.03,
            "decrement": 0.005,
            "interval": 300,
            "minimum": -0.06,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "decrement": 0.02,
            "interval": 300,
            "minimum": 0.7,
        },
    )
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 60.0,
            "hold_iterations": 500,
            "interval": 200,
            "decrement_degrees": 1.0,
        },
    )


@configclass
class ContactSweepClutterFusionV1_0CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    # 긴급 조정 : Goal Hold를 예전과 동일하게 변경

    # goal_hold_tolerance = CurriculumTermCfg(
    #     func=delayed_reward_schedule,
    #     params={
    #         "term_name": "goal_hold",
    #         "parameter": "position_tolerance",
    #         "start": 0.01,
    #         "step": -0.001,
    #         "final": 0.005,
    #         "start_iteration": 2000,
    #         "interval": 1000,
    #     },
    # )

    # 긴급 조정 : Wrecnh Change Free Force Change와 Free Moment Change를 대폭 상향 조정
    wrench_free_force_change = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_force_change",
            "start": 5000.0,
            "step": -1.0,
            "final": 10.0,
            "start_iteration": 1000,
            "interval": 250,
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_moment_change",
            "start": 500.0,
            "step": -0.1,
            "final": 1.0,
            "start_iteration": 1000,
            "interval": 250,
        },
    )

    # 긴급 조정 : Object Acceleration Free Acceleration를 대폭 상향 조정
    object_free_acceleration = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "object_acceleration",
            "parameter": "free_acceleration",
            "start": 100.0,
            "step": -0.02,
            "final": 0.1,
            "start_iteration": 1000,
            "interval": 250,
        },
    )


@configclass
class ContactSweepClutterFusionV1_1CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    obstacle_free_force = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "obstacle_contact", "parameter": "free_force",
            "initial": 100.0, "stages": ((1000, 10000, 50.0),), "interval": 250,
        },
    )
    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "initial": 100.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_force_change",
            "initial": 500.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_moment_change",
            "initial": 50.0, "stages": ((1000, 20000, 0.5),), "interval": 250,
        },
    )


@configclass
class ContactSweepClutterFusionV1_1_1CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "initial": 100.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )


@configclass
class ContactSweepClutterFusionV1_1_2CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    obstacle_free_force = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "obstacle_contact", "parameter": "free_force",
            "initial": 100.0, "stages": ((1000, 10000, 50.0),), "interval": 250,
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_force_change",
            "initial": 500.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_moment_change",
            "initial": 50.0, "stages": ((1000, 20000, 0.5),), "interval": 250,
        },
    )


@configclass
class ContactSweepClutterFusionV1_2CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    obstacle_free_force = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "obstacle_contact", "parameter": "free_force",
            "initial": 100.0, "stages": ((1000, 10000, 50.0),), "interval": 250,
        },
    )
    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "initial": 100.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_force_change",
            "initial": 500.0, "stages": ((1000, 20000, 5.0),), "interval": 250,
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_moment_change",
            "initial": 50.0, "stages": ((1000, 20000, 0.5),), "interval": 250,
        },
    )
    goal_hold_tolerance = CurriculumTermCfg(
        func=staged_reward_threshold,
        params={
            "term_name": "goal_hold", "parameter": "position_tolerance",
            "initial": 0.02, "stages": ((1000, 5000, 0.01), (5250, 10000, 0.005)), "interval": 250,
        },
    )


@configclass
class ContactSweepClutterFusionV1_3CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    obstacle_free_force = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "obstacle_contact", "parameter": "free_force",
            "points": ((1000, 200.0), (5000, 100.0), (10000, 50.0)),
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_force_change",
            "points": ((1000, 1000.0), (5000, 500.0), (10000, 100.0), (20000, 5.0)),
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_moment_change",
            "points": ((1000, 100.0), (5000, 50.0), (10000, 10.0), (20000, 0.5)),
        },
    )
    object_free_acceleration = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "points": ((2000, 100.0), (5000, 20.0), (10000, 10.0), (20000, 5.0)),
        },
    )


@configclass
class ContactSweepClutterFusionV1_3_1CurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 3,
            "final_white": 3,
            "gray_start_iteration": 4500,
            "white_start_iteration": 3000,
            "interval": 3000,
        },
    )
    # Retain the three unchanged v3 speed-threshold schedules.
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Hold through iteration 3000; the first completed interval ends at 3500.
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )

    obstacle_free_force = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "obstacle_contact", "parameter": "free_force",
            "points": ((1000, 200.0), (5000, 100.0), (10000, 50.0)),
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_force_change",
            "points": ((1000, 1000.0), (5000, 500.0), (10000, 100.0), (20000, 5.0)),
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "wrench_change", "parameter": "free_moment_change",
            "points": ((1000, 100.0), (5000, 50.0), (10000, 10.0), (20000, 0.5)),
        },
    )
    object_free_acceleration = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "object_acceleration", "parameter": "free_acceleration",
            "points": ((2000, 100.0), (5000, 20.0), (10000, 10.0), (20000, 5.0)),
        },
    )
    goal_hold_tolerance = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "goal_hold", "parameter": "position_tolerance",
            "points": ((5000, 0.02), (10000, 0.01), (20000, 0.005)),
        },
    )
    hand_contact_fade_width = CurriculumTermCfg(
        func=piecewise_linear_reward_threshold,
        params={
            "term_name": "hand_contact", "parameter": "goal_contact_fade_width",
            "points": ((5000, 0.001), (10000, 0.01)),
        },
    )


@configclass
class ContactSweepClutterFusionV1_4CurriculumCfg(ContactSweepClutterFusionV1_0CurriculumCfg):
    obstacle_population = ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.replace(
        params={**ContactSweepClutterFusionV1_0CurriculumCfg().obstacle_population.params,
                "final_gray": 5, "final_white": 5},
    )
    # Preserve the parent's first decrement at 500 and decrement per iteration.
    # Endpoints are explicit because 3300/13300/1900 are off the 250-step grid.
    arm_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "arm_speed_excess", "parameter": "free_speed",
                "initial": 1.0, "stages": ((500, 3300, 0.98, 0.7),), "interval": 250},
    )
    eef_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "eef_linear_speed", "parameter": "free_speed",
                "initial": 1.5, "stages": ((500, 13300, 1.48, 0.2),), "interval": 250},
    )
    pushing_free_speed = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "pushing_speed", "parameter": "free_speed",
                "initial": 0.1, "stages": ((500, 1900, 0.09, 0.02),), "interval": 250},
    )
    object_free_acceleration = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "object_acceleration", "parameter": "free_acceleration",
                "initial": 100.0, "stages": ((10000, 10000, 50.0, 50.0),
                                            (12000, 18000, 50.0, 10.0)), "interval": 250},
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "wrench_change", "parameter": "free_force_change",
                "initial": 5000.0, "stages": ((10000, 10000, 200.0, 200.0),
                                             (12000, 18000, 200.0, 100.0)), "interval": 250},
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "wrench_change", "parameter": "free_moment_change",
                "initial": 500.0, "stages": ((10000, 10000, 30.0, 30.0),
                                            (12000, 18000, 30.0, 20.0)), "interval": 250},
    )
    goal_hold_tolerance = CurriculumTermCfg(
        func=staged_reward_threshold_contact_sweep_clutter_fusion_quantized_ramps,
        params={"term_name": "goal_hold", "parameter": "position_tolerance",
                "initial": 0.0, "stages": ((5000, 10000, 0.01, 0.005),), "interval": 250},
    )


@configclass
class ContactSweepClutterFusionV3CurriculumCfg(ContactSweepClutterFusionCurriculumCfg):
    """
    Free Speed Curriculum
    """

    # Pushing Free Speed : 0.1 -> 0.03
    pushing_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "start": 0.1,
            "step": -0.01,
            "final": 0.03,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    # Arm Free Speed : 1.0 -> 0.7
    arm_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "start": 1.0,
            "step": -0.02,
            "final": 0.7,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    # Action Rate Weight : -0.03 -> -0.06
    action_rate_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "action_rate",
            "start": -0.03,
            "step": -0.005,
            "final": -0.06,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    # EEF Free Speed : 1.5 -> 0.3
    eef_free_speed = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "start": 1.5,
            "step": -0.02,
            "final": 0.3,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    """
    Deadzone Curriculum
    """

    # Goal Deadzone : 60 -> 0
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 60.0,
            "hold_iterations": 1000,
            "interval": 100,
            "decrement_degrees": 1.0,
        },
    )

    """
    Contact Curriculum
    """

    # Contact Any Weight : 0.4 -> 2.0
    contact_any_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "contact_any",
            "start": 0.4,
            "step": 0.02,
            "final": 2.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Contact Count Weight : 0.4 -> 2.0
    contact_count_weight = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "contact_count",
            "start": 0.4,
            "step": 0.02,
            "final": 2.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )

    """
    Wrench Curriculum
    """

    # Wrench Free Force Rate : 2000 -> 1000
    wrench_free_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_force_rate",
            "start": 2000.0,
            "step": -50.0,
            "final": 1000.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Free Moment Rate : 100 -> 50
    wrench_free_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_moment_rate",
            "start": 100.0,
            "step": -2.5,
            "final": 50.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Saturation Force Rate : 5000 -> 2500
    wrench_saturation_force_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "saturation_force_rate",
            "start": 5000.0,
            "step": -125.0,
            "final": 2500.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
    # Wrench Saturation Moment Rate : 250 -> 125.0
    wrench_saturation_moment_rate = CurriculumTermCfg(
        func=delayed_reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "saturation_moment_rate",
            "start": 250.0,
            "step": -6.25,
            "final": 125.0,
            "start_iteration": 500,
            "interval": 200,
        },
    )
