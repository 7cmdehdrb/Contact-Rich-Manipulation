"""Goal curriculum changes future resets, never an in-progress command."""

import math


def goal_deadzone_schedule(
    env, env_ids, start_degrees=60.0, hold_iterations=1000,
    interval=200, decrement_degrees=1.0,
):
    if (not math.isfinite(start_degrees) or not 0 <= start_degrees < 90
            or hold_iterations < 0 or interval <= 0
            or not math.isfinite(decrement_degrees) or decrement_degrees <= 0):
        raise ValueError("Invalid goal deadzone schedule")
    reductions = max(0, (env.learning_iteration - hold_iterations) // interval)
    degrees = max(0.0, start_degrees - decrement_degrees * reductions)
    term = env.event_manager.get_term_cfg("object_spawn")
    term.params["excluded_x_half_angle"] = math.radians(degrees)
    env.event_manager.set_term_cfg("object_spawn", term)
    return degrees
