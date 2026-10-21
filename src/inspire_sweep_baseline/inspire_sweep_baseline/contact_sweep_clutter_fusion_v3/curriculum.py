"""Absolute bounded schedules with a first update at start_iteration."""

import math


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
