"""Absolute, endpoint-exact staircase schedules with inclusive first updates."""

import math


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
