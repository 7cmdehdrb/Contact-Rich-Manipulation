"""First five-degree reduction at iteration 10000, then every 3000 iterations."""

import math


def goal_deadzone_schedule(env, env_ids, start_degrees=30.0, start_iteration=10000,
                           interval=3000, decrement_degrees=5.0):
    del env_ids
    if (not math.isfinite(start_degrees) or not 0 <= start_degrees < 90
            or start_iteration < 0 or interval <= 0
            or not math.isfinite(decrement_degrees) or decrement_degrees <= 0):
        raise ValueError('Invalid Wrench deadzone schedule')
    reductions = 0 if env.learning_iteration < start_iteration else 1 + (env.learning_iteration - start_iteration) // interval
    degrees = max(0.0, start_degrees - decrement_degrees * reductions)
    term = env.event_manager.get_term_cfg('object_spawn')
    term.params['excluded_x_half_angle'] = math.radians(degrees)
    env.event_manager.set_term_cfg('object_spawn', term)
    return degrees


def feasibility_population(iteration):
    """White first at 2000, then alternating every 3000 iterations."""
    increments = max(0, 1 + (iteration - 2000) // 3000)
    return min(4, -1 + increments // 2), min(5, -1 + (increments + 1) // 2)


def feasibility_obstacle_population_schedule(env, env_ids):
    del env_ids
    gray, white = feasibility_population(env.learning_iteration)
    term = env.event_manager.get_term_cfg('object_spawn')
    term.params.update(gray_obstacles=gray, white_obstacles=white)
    env.event_manager.set_term_cfg('object_spawn', term)
    return {'gray': gray, 'white': white}


def feasibility_layout_stages():
    """Every reachable randomized population paired with v0's direction schedule."""
    boundaries = sorted({0, *range(2000, 32001, 3000), *range(10000, 25001, 3000)})
    stages = set()
    for iteration in boundaries:
        gray, white = feasibility_population(iteration)
        reductions = 0 if iteration < 10000 else 1 + (iteration - 10000) // 3000
        degrees = max(0, 30 - 5 * reductions)
        for dg in (-1, 0, 1):
            for dw in (-1, 0, 1):
                stages.add((degrees, max(0, min(4, gray + dg)), max(0, min(5, white + dw))))
    return tuple(sorted(stages))
