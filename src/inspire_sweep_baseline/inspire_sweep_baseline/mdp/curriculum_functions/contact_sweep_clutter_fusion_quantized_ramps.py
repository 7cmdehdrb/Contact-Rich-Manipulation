"""Absolute, quantized ramps with explicit jumps and exact end iterations."""

import math as math

def staged_reward_threshold(env, env_ids, term_name, parameter, initial, stages, interval=250):
    """Stages are (start iteration, end iteration, start value, end value).

    Entering a stage may jump to its start value. Sample its linear ramp every
    interval, and always apply its exact endpoint (even off the interval grid).
    Recompute from absolute iteration so checkpoint load/rewind is stateless.
    """
    if not isinstance(interval, int) or interval <= 0 or not stages:
        raise ValueError("Expected a positive integer interval and nonempty stages")
    if not math.isfinite(initial) or initial < 0:
        raise ValueError("Expected a finite nonnegative initial threshold")
    previous_end = -1
    for start, end, first, last in stages:
        if (not isinstance(start, int) or not isinstance(end, int)
                or start <= previous_end or end < start
                or not all(math.isfinite(v) and v >= 0 for v in (first, last))
                or (start == end and first != last)):
            raise ValueError("Expected ordered, nonoverlapping stages with nonnegative thresholds")
        previous_end = end
    value = initial
    iteration = env.learning_iteration
    for start, end, first, last in stages:
        if iteration < start:
            break
        if iteration >= end:
            value = last
        else:
            elapsed = ((iteration - start) // interval) * interval
            value = first + (last - first) * elapsed / (end - start)
            break
    term = env.reward_manager.get_term_cfg(term_name)
    term.params[parameter] = value
    env.reward_manager.set_term_cfg(term_name, term)
    return value
