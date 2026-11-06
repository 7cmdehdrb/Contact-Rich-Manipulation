"""Absolute-iteration termination schedules for Relative environments."""

import math


def arm_velocity_threshold_schedule(env, env_ids, initial=4.0, minimum=2.0,
                                    start_iteration=10000, interval=250, decrement=0.05):
    """Hold through start_iteration, then lower the cutoff once per interval."""
    del env_ids
    if (not all(math.isfinite(value) for value in (initial, minimum, decrement))
            or not 0 < minimum <= initial or decrement <= 0
            or start_iteration < 0 or interval <= 0):
        raise ValueError('Invalid Relative arm velocity termination schedule')
    reductions = max(0, (env.learning_iteration - start_iteration) // interval)
    threshold = max(minimum, round(initial - reductions * decrement, 10))
    term = env.termination_manager.get_term_cfg('arm_joint_velocity')
    term.params['threshold'] = threshold
    env.termination_manager.set_term_cfg('arm_joint_velocity', term)
    return threshold
