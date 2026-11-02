"""Explicit v0_2 schedules; goal radius and Gray padding are fixed in Revision."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..curriculum_functions.contact_sweep_clutter_fusion import goal_deadzone_schedule, obstacle_population_schedule
from ..curriculum_functions.contact_sweep_clutter_fusion_holding import reward_schedule


def _schedule(term_name, parameter, points, interval=250):
    return CurriculumTermCfg(func=reward_schedule, params={
        "term_name": term_name, "parameter": parameter, "points": points, "interval": interval,
    })


@configclass
class RevisionCurriculumCfg:
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 1, "initial_white": 1, "final_gray": 4, "final_white": 5,
            "gray_start_iteration": 2500, "white_start_iteration": 2000, "interval": 1000,
        },
    )
    target_free_speed = _schedule("target_speed_penalty", "free_speed",
        ((0, 0.1), (499, 0.1), (500, 0.09), (1900, 0.02)))
    object_free_acceleration = _schedule("object_penalty", "free_acceleration",
        ((3000, 50.0), (10000, 20.0)), interval=1)
    object_acceleration_weight = _schedule("object_penalty", "acceleration_weight",
        ((3000, 0.0), (10000, -0.05)), interval=1)
    target_goal_direction_deadzone = CurriculumTermCfg(
        func=goal_deadzone_schedule,
        params={"start_degrees": 30.0, "hold_iterations": 3000, "interval": 500, "decrement_degrees": 5.0},
    )
    wrist_wrench_free_force_change = _schedule("wrist_wrench_change_penalty", "free_force_change",
        ((3000, 50.0), (10000, 20.0)))
    wrist_wrench_free_moment_change = _schedule("wrist_wrench_change_penalty", "free_moment_change",
        ((3000, 5.0), (10000, 2.0)))
    wrist_wrench_change_weight = _schedule("wrist_wrench_change_penalty", None,
        ((0, 0.0), (1000, 0.0), (3000, -0.03), (6000, -0.1)))
    obstacle_contact_free_force = _schedule("obstacle_contact_penalty", "free_force",
        ((0, 80.0), (3000, 80.0), (6000, 50.0), (10000, 50.0), (12000, 30.0)))
    obstacle_contact_weight = _schedule("obstacle_contact_penalty", None,
        ((0, 0.0), (1000, 0.0), (3000, -0.02), (6000, -0.05), (10000, -0.05), (12000, -0.1)))


@configclass
class RevisionV1CurriculumCfg(RevisionCurriculumCfg):
    obstacle_population = CurriculumTermCfg(
        func=obstacle_population_schedule,
        params={
            "initial_gray": 3, "initial_white": 3, "final_gray": 4, "final_white": 5,
            "gray_start_iteration": 12000, "white_start_iteration": 10000, "interval": 4000,
        },
    )
    object_acceleration_weight = None
    object_free_acceleration = None
    obstacle_contact_free_force = None
    obstacle_contact_weight = None
    wrist_wrench_change_weight = None
    wrist_wrench_free_force_change = None
    wrist_wrench_free_moment_change = None
