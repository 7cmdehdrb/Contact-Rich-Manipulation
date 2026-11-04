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
