"""Iteration-based linear interpolation between explicit threshold points."""

import math


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
