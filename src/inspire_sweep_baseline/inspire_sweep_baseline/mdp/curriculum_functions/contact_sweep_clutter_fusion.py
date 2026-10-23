"""Goal curriculum changes future resets, never an in-progress command.

Episode-local active obstacle slots and absolute population schedules.

Absolute, endpoint-exact staircase schedules with inclusive first updates.

Iteration-based linear interpolation between explicit threshold points.

Absolute bounded schedules with a first update at start_iteration."""

import math as math

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


def obstacle_population_schedule(
    env, env_ids, initial_gray=1, initial_white=1, final_gray=3, final_white=3,
    gray_start_iteration=4500, white_start_iteration=3000, interval=3000,
):
    """Update future reset counts; never change an ongoing episode's mask."""
    del env_ids
    values = (initial_gray, initial_white, final_gray, final_white, gray_start_iteration, white_start_iteration)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("Population counts and schedule start iterations must be nonnegative integers")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("Population interval must be a positive integer")
    if final_gray < initial_gray or final_white < initial_white:
        raise ValueError("Population final counts must not be smaller than initial counts")
    iteration = env.learning_iteration
    counts = {}
    for role, initial, final, start in (
        ("gray", initial_gray, final_gray, gray_start_iteration),
        ("white", initial_white, final_white, white_start_iteration),
    ):
        increments = 0 if iteration < start else (iteration - start) // interval + 1
        counts[role + "_obstacles"] = min(initial + increments, final, getattr(env.cfg, "max_" + role + "_obstacles"))
    term = env.event_manager.get_term_cfg("object_spawn")
    term.params.update(counts)
    env.event_manager.set_term_cfg("object_spawn", term)
    return counts


def staged_reward_threshold(env, env_ids, term_name, parameter, initial, stages, interval=250):
    """Apply equal decrements at start, start+interval, ..., end of each stage.

    Recompute from the absolute learning iteration, so reset, resume and rewind
    need no mutable schedule history. Subsequent stages start after the previous
    endpoint; values hold between stages and after the final endpoint.
    """
    if not isinstance(interval, int) or interval <= 0 or not math.isfinite(initial) or not stages:
        raise ValueError("Expected a positive integer interval, finite initial and nonempty stages")
    previous_end = -1
    previous_value = initial
    value = initial
    for start, end, final in stages:
        if (not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or start <= previous_end or end < start
                or (end - start) % interval or not math.isfinite(final)
                or final > previous_value):
            raise ValueError("Expected ordered, aligned decreasing stages with finite endpoints")
        count = (end - start) // interval + 1
        updates = max(0, min(count, (env.learning_iteration - start) // interval + 1))
        if updates:
            value = final if updates == count else previous_value + (final - previous_value) * updates / count
        previous_end, previous_value = end, final
    term = env.reward_manager.get_term_cfg(term_name)
    term.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, term)
    return value


def piecewise_linear_reward_threshold(env, env_ids, term_name, parameter, points):
    """Hold before/at the first point, interpolate, then hold the final value.

    Points are (absolute iteration, threshold) pairs. Every learning iteration
    updates the value; there is no interval quantization or mutable history.
    Full/partial reset, resume and rewind therefore give the same threshold.
    """
    if not points:
        raise ValueError("Expected at least one threshold point")
    previous_iteration = -1
    for iteration, value in points:
        if (not isinstance(iteration, int) or iteration <= previous_iteration
                or not math.isfinite(value) or value < 0):
            raise ValueError("Expected increasing nonnegative integer iterations and finite nonnegative values")
        previous_iteration = iteration

    iteration = env.learning_iteration
    value = points[0][1]
    for (left, start), (right, end) in zip(points, points[1:]):
        if iteration <= left:
            break
        if iteration >= right:
            value = end
        else:
            value = start + (end - start) * ((iteration - left) / (right - left))
            break
    term = env.reward_manager.get_term_cfg(term_name)
    term.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, term)
    return value


def delayed_reward_schedule(
    env, env_ids, term_name, start, step, final, interval=100,
    start_iteration=500, parameter=None,
):
    """Recompute from iteration, including on checkpoint rewind and reset."""
    if (interval <= 0 or start_iteration < 0
            or not all(math.isfinite(value) for value in (start, step, final))
            or step == 0 or (final - start) * step < 0):
        raise ValueError("Expected a finite step toward final and valid iteration bounds")
    updates = max(0, (env.learning_iteration - start_iteration) // interval + 1)
    value = round(start + step * updates, 12)
    value = min(final, value) if step > 0 else max(final, value)
    term = env.reward_manager.get_term_cfg(term_name)
    if parameter is None:
        term.weight = value
    else:
        term.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, term)
    return value
