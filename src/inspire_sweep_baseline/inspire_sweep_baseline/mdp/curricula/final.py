"""Explicit Final v0 terms, frozen from Holding OSC ObjectRandom v3."""

from ..curriculum_functions.contact_sweep_clutter_fusion_holding import reward_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import goal_deadzone_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion import (
    obstacle_population_schedule,
)
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..curriculum_functions.contact_sweep_clutter_fusion_holding_osc_object_random import (
    gray_zone_padding_schedule,
)


@configclass
class FinalCurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1,
            "initial_white": 1,
            "final_gray": 6,
            "final_white": 6,
            "gray_start_iteration": 4000,
            "white_start_iteration": 3000,
            "interval": 2000,
        },
    )
    pushing_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "pushing_speed",
            "parameter": "free_speed",
            "points": ((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)),
            "interval": 250,
        },
    )
    arm_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "arm_speed_excess",
            "parameter": "free_speed",
            "points": ((0, 1.0), (499, 1.0), (500, 0.98), (3300, 0.7)),
            "interval": 250,
        },
    )
    eef_free_speed = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "eef_linear_speed",
            "parameter": "free_speed",
            "points": ((0, 1.5), (499, 1.5), (500, 1.48), (13300, 0.2)),
            "interval": 250,
        },
    )
    goal_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={
            "start_degrees": 30.0,
            "hold_iterations": 3000,
            "interval": 500,
            "decrement_degrees": 5.0,
        },
    )
    wrench_free_force_change = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_force_change",
            "points": (
                (0, 50.0),
                (3000, 50.0),
                (6000, 30.0),
                (10000, 30.0),
                (12000, 20.0),
            ),
            "interval": 250,
        },
    )
    wrench_free_moment_change = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": "free_moment_change",
            "points": ((0, 5.0), (3000, 5.0), (6000, 3.0), (10000, 3.0), (12000, 2.0)),
            "interval": 250,
        },
    )
    object_free_acceleration = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "object_acceleration",
            "parameter": "free_acceleration",
            "points": (
                (0, 50.0),
                (3000, 50.0),
                (6000, 30.0),
                (10000, 30.0),
                (12000, 20.0),
            ),
            "interval": 250,
        },
    )
    obstacle_free_force = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "obstacle_contact",
            "parameter": "free_force",
            "points": (
                (0, 80.0),
                (3000, 80.0),
                (6000, 50.0),
                (10000, 50.0),
                (12000, 30.0),
            ),
            "interval": 250,
        },
    )
    acceleration_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "object_acceleration",
            "parameter": None,
            "points": (
                (0, 0.0),
                (1000, 0.0),
                (3000, -0.01),
                (6000, -0.03),
                (10000, -0.03),
                (12000, -0.05),
            ),
            "interval": 250,
        },
    )
    wrench_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "wrench_change",
            "parameter": None,
            "points": ((0, 0.0), (1000, 0.0), (3000, -0.03), (6000, -0.1)),
            "interval": 250,
        },
    )
    obstacle_weight = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "obstacle_contact",
            "parameter": None,
            "points": (
                (0, 0.0),
                (1000, 0.0),
                (3000, -0.02),
                (6000, -0.05),
                (10000, -0.05),
                (12000, -0.1),
            ),
            "interval": 250,
        },
    )
    goal_position_scale = None
    goal_speed_scale = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "goal_hold",
            "parameter": "speed_scale",
            "points": ((0, 0.05), (6000, 0.05), (10000, 0.02)),
            "interval": 250,
        },
    )
    goal_position_tolerance = CurriculumTermCfg(
        func=reward_schedule,
        params={
            "term_name": "goal_hold",
            "parameter": "position_tolerance",
            "points": ((0, 0.02), (5000, 0.02), (10000, 0.01), (20000, 0.005)),
            "interval": 1,
        },
    )
    gray_zone_padding = CurriculumTermCfg(
        func=gray_zone_padding_schedule,
        params={"points": ((6000, 0.0), (12000, 0.15)), "interval": 250},
    )


def _final_schedule(term_name, parameter, points, interval=250):
    return CurriculumTermCfg(func=reward_schedule, params={
        'term_name': term_name, 'parameter': parameter, 'points': points, 'interval': interval,
    })


@configclass
class FinalV1CurriculumCfg:
    obstacle_population = CurriculumTermCfg(func=obstacle_population_schedule, params={
        'initial_gray': 1, 'initial_white': 1, 'final_gray': 8, 'final_white': 8,
        'gray_start_iteration': 3000, 'white_start_iteration': 2000, 'interval': 2000,
    })
    obstacle_gray_zone_padding = CurriculumTermCfg(func=gray_zone_padding_schedule, params={
        'points': ((4000, 0.0), (10000, 0.15)), 'interval': 1000,
    })
    object_free_speed = _final_schedule('object_speed_penalty', 'free_speed',
        ((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)))
    object_free_acceleration = _final_schedule('object_penalty', 'free_acceleration',
        ((3000, 50.0), (10000, 20.0)), interval=1)
    object_acceleration_weight = _final_schedule('object_penalty', 'acceleration_weight',
        ((3000, 0.0), (10000, -0.05)), interval=1)
    goal_deadzone = FinalCurriculumCfg().goal_deadzone
    goal_speed_scale = FinalCurriculumCfg().goal_speed_scale
    goal_position_tolerance = _final_schedule('goal_hold', 'position_tolerance',
        ((4000, 0.02), (10000, 0.01)), interval=1)
    wrench_free_force_change = _final_schedule('wrench_change_penalty', 'free_force_change',
        ((3000, 50.0), (10000, 20.0)))
    wrench_free_moment_change = _final_schedule('wrench_change_penalty', 'free_moment_change',
        ((3000, 5.0), (10000, 2.0)))
    wrench_weight = _final_schedule('wrench_change_penalty', None,
        FinalCurriculumCfg().wrench_weight.params['points'])
    obstacle_free_force = _final_schedule('obstacle_contact_penalty', 'free_force',
        FinalCurriculumCfg().obstacle_free_force.params['points'])
    obstacle_weight = _final_schedule('obstacle_contact_penalty', None,
        FinalCurriculumCfg().obstacle_weight.params['points'])


@configclass
class FinalV1TestCurriculumCfg(FinalV1CurriculumCfg):
    """Apply v1's acceleration schedule to its standalone reward term."""

    object_free_acceleration = _final_schedule('object_acceleration_penalty', 'free_acceleration',
        FinalV1CurriculumCfg().object_free_acceleration.params['points'], interval=1)
    object_acceleration_weight = _final_schedule('object_acceleration_penalty', None,
        FinalV1CurriculumCfg().object_acceleration_weight.params['points'], interval=1)


@configclass
class FinalGoalAccuracyCurriculumCfg(FinalV1CurriculumCfg):
    obstacle_population = CurriculumTermCfg(func=obstacle_population_schedule, params={
        'initial_gray': 1, 'initial_white': 1, 'final_gray': 8, 'final_white': 8,
        'gray_start_iteration': 2500, 'white_start_iteration': 2000, 'interval': 1000,
    })
    object_free_speed = None
    goal_speed_scale = None
    goal_position_tolerance = None
